"""Crawler that discovers all product handles from Porter James collections.

Uses the Shopify JSON API to fetch collections and their products.
No browser automation needed — Shopify exposes clean JSON endpoints.
"""

import time
import requests
from typing import Dict, List, Optional, Set, Tuple

import config

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
)


def fetch_json(url: str, params: Optional[Dict] = None, retries: int = config.MAX_RETRIES) -> Optional[dict]:
    """Fetch a JSON URL with retry logic."""
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [retry {attempt + 1}/{retries}] {url} failed: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] Failed to fetch {url} after {retries} attempts: {e}")
                return None


def get_collections() -> List[Dict]:
    """Fetch all collections from the store."""
    data = fetch_json(config.COLLECTIONS_URL)
    if data is None:
        print("[ERROR] Could not fetch collections list.")
        return []
    collections = data.get("collections", [])
    print(f"[crawler] Found {len(collections)} collections on the store.")
    return collections


def get_products_from_collection(collection_handle: str) -> List[Dict]:
    """Fetch all products from a specific collection, handling pagination.

    Shopify's products.json endpoint paginates with ?page=N&limit=250.
    """
    all_products: List[Dict] = []
    page = 1
    total_fetched = 0

    while True:
        url = f"{config.BASE_URL}/collections/{collection_handle}/products.json"
        params = {"limit": config.SHOPIFY_PAGE_LIMIT, "page": page}
        data = fetch_json(url, params=params)

        if data is None:
            break

        products = data.get("products", [])
        if not products:
            break

        all_products.extend(products)
        total_fetched += len(products)
        print(f"  [crawler] Page {page}: got {len(products)} products from '{collection_handle}' (total: {total_fetched})")

        # If we got fewer than the limit, we're on the last page
        if len(products) < config.SHOPIFY_PAGE_LIMIT:
            break

        page += 1
        time.sleep(0.3)  # be polite

    return all_products


def crawl_all_categories(category_handles: Optional[List[str]] = None) -> Tuple[List[Tuple[str, Dict]], Dict[str, List[str]]]:
    """Crawl all specified categories.

    Returns:
        Tuple of:
        - List of (primary_category, product_dict) for each unique product
        - Dict mapping product handle → list of category handles it belongs to
    """
    if category_handles is None:
        category_handles = config.CATEGORY_HANDLES

    all_products: List[Tuple[str, Dict]] = []

    for handle in category_handles:
        print(f"\n[crawler] Scraping category: '{handle}' ({config.CATEGORY_MAP.get(handle, handle)})")
        products = get_products_from_collection(handle)
        for product in products:
            all_products.append((handle, product))
        print(f"  [crawler] Done: {len(products)} products from '{handle}'")
        time.sleep(0.5)  # be polite between categories

    print(f"\n[crawler] TOTAL: {len(all_products)} product entries across {len(category_handles)} categories.")

    # Deduplicate by product handle (keep all category assignments)
    unique_handles: Set[str] = set()
    deduped: List[Tuple[str, Dict]] = []
    categories_per_handle: Dict[str, List[str]] = {}

    for cat_handle, product in all_products:
        p_handle = product.get("handle", "")
        if p_handle not in categories_per_handle:
            categories_per_handle[p_handle] = []
        categories_per_handle[p_handle].append(cat_handle)

        if p_handle not in unique_handles:
            unique_handles.add(p_handle)
            deduped.append((product.get("product_type", config.CATEGORY_MAP.get(cat_handle, cat_handle)), product))

    print(f"[crawler] Unique products: {len(deduped)}")
    return deduped, categories_per_handle


if __name__ == "__main__":
    products, categories = crawl_all_categories()
    for cat, prod in products[:5]:
        print(f"  - {prod.get('title')} [{cat}]")
