"""Scraper that fetches individual product details and transforms data into the DB schema.

Uses Shopify's product.json endpoint to get detailed product data including
variants (sizes, prices), images, and metadata.
"""

import json
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

import config

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
)


def clean_html(html_text: str) -> str:
    """Strip HTML tags and clean whitespace from a string."""
    if not html_text:
        return ""
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_product_json(handle: str) -> Optional[Dict]:
    """Fetch the full JSON data for a single product by its handle.

    Shopify exposes product data at /products/{handle}.json
    Uses exponential backoff with special handling for 429 rate limits
    (checks Retry-After header, longer waits).
    """
    url = f"{config.BASE_URL}/products/{handle}.json"
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = session.get(url, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("product")
        except requests.RequestException as e:
            is_rate_limit = isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 429

            if attempt < config.MAX_RETRIES - 1:
                if is_rate_limit:
                    # Try Retry-After header first, fall back to longer backoff
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            wait = config.RATE_LIMIT_429_DELAY * (2 ** attempt)
                    else:
                        wait = config.RATE_LIMIT_429_DELAY * (2 ** attempt)
                else:
                    wait = 2 ** attempt

                print(f"  [retry {attempt + 1}/{config.MAX_RETRIES}] {url} failed: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] Failed to fetch {url} after {config.MAX_RETRIES} attempts: {e}")
                return None


def format_price(amount: str, currency: str) -> str:
    """Format a price as 'amountCURRENCY' e.g. '5400.00CZK'."""
    try:
        amt = float(amount)
        if amt == int(amt):
            return f"{int(amt)}{currency}"
        return f"{amt:.2f}{currency}"
    except (ValueError, TypeError):
        return f"{amount}{currency}"


def fetch_exchange_rates() -> Dict[str, float]:
    """Fetch current exchange rates from NZD to EUR and USD.

    Falls back to approximate hardcoded rates if the API is unavailable.
    """
    fallback_rates = {"EUR": 0.55, "USD": 0.59}  # approximate NZD rates
    try:
        resp = requests.get(config.EXCHANGE_RATE_API, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates", {})
        result = {}
        if "EUR" in rates:
            result["EUR"] = float(rates["EUR"])
        if "USD" in rates:
            result["USD"] = float(rates["USD"])
        if result:
            print(f"[scraper] Exchange rates (1 NZD → EUR: {result.get('EUR', 'N/A')}, USD: {result.get('USD', 'N/A')})")
            return result
    except Exception as e:
        print(f"[scraper] Warning: Could not fetch exchange rates: {e}")

    print(f"[scraper] Using fallback rates: EUR={fallback_rates['EUR']}, USD={fallback_rates['USD']}")
    return fallback_rates


def transform_product(
    product: Dict,
    primary_category: str,
    all_categories: List[str],
    exchange_rates: Dict[str, float],
) -> Dict:
    """Transform a Shopify product JSON dict into our database schema.

    Args:
        product: The Shopify product dict from /products/{handle}.json
        primary_category: The primary category label for this product
        all_categories: All category labels this product belongs to
        exchange_rates: Exchange rates from CZK to other currencies

    Returns:
        A dict ready for upsert into the Supabase 'products' table.
    """
    handle = product.get("handle", "")
    title = product.get("title", "")
    product_id = str(product.get("id", handle))
    body_html = product.get("body_html", "") or ""
    description = clean_html(body_html)
    vendor = product.get("vendor", config.BRAND)
    tags_raw = product.get("tags", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

    # --- Images ---
    images = product.get("images", [])
    main_image_url = ""
    additional_image_urls = []

    for i, img in enumerate(images):
        src = img.get("src", "")
        if src:
            if i == 0:
                main_image_url = src
            else:
                additional_image_urls.append(src)

    additional_images_str = " , ".join(additional_image_urls) if additional_image_urls else None

    # --- Variants (sizes, prices) ---
    variants = product.get("variants", [])
    # Collect unique sizes/options
    options = product.get("options", [])
    # Find which option index corresponds to "Size" (could be option1, option2, or option3)
    size_option_index = None
    for opt in options:
        opt_name = opt.get("name", "").lower()
        if opt_name == "size":
            size_option_index = options.index(opt)
            break

    size_values = []
    for v in variants:
        # Get the value from the appropriate optionX field
        if size_option_index is not None:
            opt_key = f"option{size_option_index + 1}"
            val = v.get(opt_key, "")
        else:
            # Fallback: try option1, then title
            val = v.get("option1", "") or v.get("title", "")
        if val and val != "Default Title" and val not in size_values:
            size_values.append(val)

    size_str = ", ".join(size_values) if size_values else None

    # --- Price logic ---
    # Shopify: price = current selling price, compare_at_price = original price (when on sale)
    # If compare_at_price is set, the product is ON SALE.
    # We want: price column = original price (no sale), sale column = sale price (or null)
    store_currency = "NZD"  # Porter James uses NZD (New Zealand Dollar)

    # Get the first available variant's prices
    # We'll use the minimum price variant as the "starting from" price
    min_price = None
    min_compare_at = None

    for v in variants:
        v_price = v.get("price")
        v_compare = v.get("compare_at_price")
        try:
            p = float(v_price) if v_price else None
            if p is not None and (min_price is None or p < min_price):
                min_price = p
                min_compare_at = float(v_compare) if v_compare else None
        except (ValueError, TypeError):
            continue

    if min_price is not None:
        if min_compare_at and min_compare_at > min_price:
            # Product is on sale: compare_at_price is the original
            original_price = min_compare_at
            sale_price = min_price
        else:
            original_price = min_price
            sale_price = None
    else:
        original_price = None
        sale_price = None

    # Format prices with currency codes
    price_parts = []

    if original_price is not None:
        # User wants EUR first (highest priority), then USD, then store currency
        if exchange_rates:
            if "EUR" in exchange_rates:
                eur_price = original_price * exchange_rates["EUR"]
                price_parts.append(format_price(f"{eur_price:.2f}", "EUR"))
            if "USD" in exchange_rates:
                usd_price = original_price * exchange_rates["USD"]
                price_parts.append(format_price(f"{usd_price:.2f}", "USD"))
        # Add NZD last (store's actual currency)
        price_parts.append(format_price(str(original_price), "NZD"))

    price_str = ", ".join(price_parts) if price_parts else None

    sale_str = None
    if sale_price is not None:
        sale_parts = []
        if exchange_rates:
            if "EUR" in exchange_rates:
                sale_parts.append(format_price(f"{sale_price * exchange_rates['EUR']:.2f}", "EUR"))
            if "USD" in exchange_rates:
                sale_parts.append(format_price(f"{sale_price * exchange_rates['USD']:.2f}", "USD"))
        sale_parts.append(format_price(str(sale_price), "NZD"))
        sale_str = ", ".join(sale_parts)

    # --- Category ---
    category_str = ", ".join(all_categories) if len(all_categories) > 1 else (all_categories[0] if all_categories else None)

    # --- Metadata (all info in one place) ---
    metadata = {
        "shopify_id": product.get("id"),
        "handle": handle,
        "vendor": vendor,
        "product_type": product.get("product_type"),
        "published_at": product.get("published_at"),
        "created_at": product.get("created_at"),
        "updated_at": product.get("updated_at"),
        "tags": tags,
        "options": options,
        "variants": [
            {
                "id": v.get("id"),
                "title": v.get("title"),
                "sku": v.get("sku"),
                "price": v.get("price"),
                "compare_at_price": v.get("compare_at_price"),
                "available": v.get("available"),
                "inventory_quantity": v.get("inventory_quantity"),
            }
            for v in variants
        ],
        "total_variants": len(variants),
        "total_images": len(images),
        "currency": store_currency,
    }

    # --- Product URL ---
    product_url = f"{config.BASE_URL}/products/{handle}"

    # --- Text for info_embedding (built here, used by embedder) ---
    info_text_parts = [
        f"Title: {title}",
        f"Description: {description}",
        f"Price: {price_str}" if price_str else "",
        f"Sale: {sale_str}" if sale_str else "",
        f"Category: {category_str}" if category_str else "",
        f"Gender: {config.GENDER}",
        f"Brand: {config.BRAND}",
        f"Tags: {', '.join(tags)}" if tags else "",
        f"Sizes: {size_str}" if size_str else "",
    ]
    info_text = ". ".join(p for p in info_text_parts if p)

    now = datetime.now(timezone.utc).isoformat()

    return {
        "id": product_id,
        "source": config.SOURCE,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": main_image_url if main_image_url else None,  # NOT NULL constraint
        "brand": config.BRAND,
        "title": title,
        "description": description,
        "category": category_str,
        "gender": config.GENDER,
        "search_tsv": None,  # handled by DB trigger or we can compute
        "created_at": now,
        "metadata": json.dumps(metadata),
        "size": size_str,
        "second_hand": config.SECOND_HAND,
        "image_embedding": None,  # filled later by embedder
        "country": None,  # always NULL
        "compressed_image_url": None,
        "tags": tags,
        "search_vector": None,
        "title_tsv": None,
        "brand_tsv": None,
        "description_tsv": None,
        "other": None,
        "price": price_str,
        "sale": sale_str,
        "additional_images": additional_images_str,
        "info_embedding": None,  # filled later by embedder
        "_info_text": info_text,  # internal: used by embedder to avoid re-building
    }


def scrape_all_products(
    category_product_pairs: List[Tuple[str, Dict]],
    categories_per_handle: Dict[str, List[str]],
    exchange_rates: Dict[str, float],
) -> List[Dict]:
    """Scrape detailed data for all unique products.

    Args:
        category_product_pairs: List of (category_handle, product_summary_dict) from crawler
        categories_per_handle: Mapping of product handle → list of category handles
        exchange_rates: Currency exchange rates

    Returns:
        List of transformed product dicts ready for embedding + DB insertion
    """
    # Deduplicate by product handle
    seen_handles = set()
    products_to_scrape = []

    for primary_cat, product_summary in category_product_pairs:
        handle = product_summary.get("handle", "")
        if handle not in seen_handles:
            seen_handles.add(handle)
            # Get all categories this product belongs to
            cat_handles = categories_per_handle.get(handle, [primary_cat])
            all_category_labels = []
            for ch in cat_handles:
                label = config.CATEGORY_MAP.get(ch, ch)
                if label not in all_category_labels:
                    all_category_labels.append(label)

            # Use product_type from Shopify as primary if available
            shopify_type = product_summary.get("product_type", "")
            primary = shopify_type if shopify_type else all_category_labels[0] if all_category_labels else primary_cat

            products_to_scrape.append((handle, primary, all_category_labels))

    print(f"\n[scraper] Fetching details for {len(products_to_scrape)} unique products...")

    results = []
    for i, (handle, primary_cat, all_cats) in enumerate(products_to_scrape, 1):
        print(f"  [{i}/{len(products_to_scrape)}] Fetching: {handle}...")

        product_data = fetch_product_json(handle)
        if product_data is None:
            print(f"  [SKIP] Could not fetch product data for '{handle}'")
            continue

        transformed = transform_product(product_data, primary_cat, all_cats, exchange_rates)
        results.append(transformed)
        print(f"    → {transformed['title']} | {transformed['price'] or 'N/A'}")

        # Be polite — rate limit between requests
        time.sleep(config.RATE_LIMIT_DELAY)

    print(f"\n[scraper] Successfully scraped {len(results)} products.")
    return results


if __name__ == "__main__":
    # Quick test
    from crawler import crawl_all_categories
    pairs, cat_map = crawl_all_categories()
    rates = fetch_exchange_rates()
    products = scrape_all_products(pairs, cat_map, rates)
    if products:
        p = products[0]
        print(f"\nSample product:")
        print(f"  Title: {p['title']}")
        print(f"  Price: {p['price']}")
        print(f"  Sale: {p['sale']}")
        print(f"  Category: {p['category']}")
        print(f"  Images: {p['image_url'][:80]}...")
        print(f"  Additional: {str(p['additional_images'])[:80]}...")
