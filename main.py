#!/usr/bin/env python3
"""
Porter James Product Scraper — Smart Orchestrator

Full pipeline:
1. Crawl all categories → discover all product handles
2. Scrape each product's details via Shopify JSON API
3. Compare scraped data against existing DB records
4. Classify: new / updated / unchanged
5. Generate embeddings ONLY for new + image-changed products
6. Batch upsert (50/batch, 3 retries) new + updated products
7. Remove stale products (missed for 2+ consecutive runs)
8. Print detailed run summary

Usage:
    python main.py                          # Full smart run
    python main.py --skip-embeddings        # Skip embedding generation
    python main.py --skip-db                # Scrape + classify only
    python main.py --force                  # Re-process all products (ignore cache)
    python main.py --skip-stale             # Don't delete stale products
    python main.py --max-products 10        # Limit for testing
    python main.py --verify-db              # Just check DB connection
    python main.py --help                   # Show usage
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple

from crawler import crawl_all_categories
from scraper import fetch_exchange_rates, scrape_all_products
from embedder import SigLipEmbedder, add_embeddings_to_products
from database import Database, verify_connection, compute_change_hash
import config


# ---------------------------------------------------------------------------
# State file management — tracks seen products across runs for staleness
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load scraper state from local JSON file."""
    if os.path.exists(config.STATE_FILE):
        try:
            with open(config.STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "last_run_urls": [],
        "strike_urls": {},       # product_url → consecutive_misses
        "last_run_timestamp": None,
    }


def save_state(
    seen_urls: Set[str],
    previous_run_urls: Set[str],
    previous_strikes: Dict[str, int],
):
    """Update strikes and persist state.

    - Products seen in current run → reset strike to 0.
    - Products in previous run but NOT in current run → strike += 1.
    - Products neither seen nor in previous run → keep existing strikes.
    """
    current_strikes: Dict[str, int] = {}

    # Reset strikes for products seen this run
    for url in seen_urls:
        current_strikes[url] = 0

    # Increment strikes for products that disappeared
    for url in previous_run_urls:
        if url not in seen_urls:
            prev = previous_strikes.get(url, 0)
            current_strikes[url] = prev + 1

    # Carry over strikes for products that were already absent
    for url, strike in previous_strikes.items():
        if url not in current_strikes and url not in seen_urls:
            current_strikes[url] = strike + 1  # they missed another run

    state = {
        "last_run_urls": list(seen_urls),
        "strike_urls": current_strikes,
        "last_run_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(config.STATE_FILE), exist_ok=True)
    with open(config.STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_stale_urls(state: dict, seen_urls: Set[str]) -> Set[str]:
    """Get URLs that have been missing for STALE_THRESHOLD_RUNS or more."""
    strikes = state.get("strike_urls", {})
    stale = set()
    for url, strike_count in strikes.items():
        if url not in seen_urls and strike_count >= config.STALE_THRESHOLD_RUNS:
            stale.add(url)
    return stale


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_products(
    scraped: List[Dict],
    existing: Dict[str, Dict],
    force: bool = False,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Classify scraped products as new, updated, or unchanged vs existing DB records.

    Args:
        scraped: List of freshly scraped product dicts.
        existing: Dict of product_url → existing DB product dict.
        force: If True, all products are treated as new (re-process everything).

    Returns:
        Tuple of (new_products, updated_products, unchanged_product_urls).
    """
    new: List[Dict] = []
    updated: List[Dict] = []
    unchanged_urls: List[str] = []

    for p in scraped:
        url = p.get("product_url", "")
        existing_product = existing.get(url)

        if not existing_product or force:
            # Brand new product → needs full processing
            p["_needs_embedding"] = True
            p["_change_hash"] = compute_change_hash(p)
            p["_classification"] = "new"
            new.append(p)
            continue

        # Exists — compare change hashes
        scraped_hash = compute_change_hash(p)
        stored_hash = existing_product.get("_stored_hash")

        # Also check if image changed (triggers re-embedding)
        existing_image = existing_product.get("image_url", "")
        new_image = p.get("image_url", "")
        image_changed = existing_image != new_image

        if stored_hash == scraped_hash and not image_changed:
            # Exactly the same → skip entirely
            p["_classification"] = "unchanged"
            # Carry forward existing embeddings if present
            if existing_product.get("image_embedding"):
                p["image_embedding"] = existing_product["image_embedding"]
            if existing_product.get("info_embedding"):
                p["info_embedding"] = existing_product["info_embedding"]
            unchanged_urls.append(url)
        else:
            # Something changed → needs new embeddings
            p["_needs_embedding"] = True
            p["_change_hash"] = compute_change_hash(p)  # recompute with all fields
            p["_classification"] = "updated"
            updated.append(p)

    return new, updated, unchanged_urls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Porter James Product Scraper — Smart orchestrator with "
        "batch upserts, change detection, and stale product cleanup."
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip image and text embedding generation",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip all database operations (for testing scrape only)",
    )
    parser.add_argument(
        "--verify-db",
        action="store_true",
        help="Only verify database connection and exit",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=0,
        help="Maximum number of products to scrape (0 = all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of all products (ignore change detection)",
    )
    parser.add_argument(
        "--skip-stale",
        action="store_true",
        help="Skip stale product deletion",
    )
    return parser.parse_args()


def print_header(step: str):
    width = 72
    print()
    print("=" * width)
    print(f"  STEP: {step}")
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print()
    print("=" * 72)
    print("  PORTER JAMES PRODUCT SCRAPER  (smart mode)")
    print(f"  Started: {timestamp}")
    print("=" * 72)

    # --- Verify DB connection early if not skipping DB ---
    if not args.skip_db:
        print_header("Verifying database connection")
        if not verify_connection():
            print("[ERROR] Could not connect to Supabase. Check your credentials.")
            sys.exit(1)

    if args.verify_db:
        print("[main] DB connection verified. Exiting.")
        return

    # --- Step 1: Crawl categories ---
    print_header("Crawling categories to discover all products")
    category_product_pairs, categories_per_handle = crawl_all_categories()

    if not category_product_pairs:
        print("[ERROR] No products found. Exiting.")
        sys.exit(1)

    if args.max_products > 0:
        category_product_pairs = category_product_pairs[: args.max_products]
        print(f"[main] Limited to {args.max_products} products (--max-products).")

    # --- Step 2: Scrape product details ---
    print_header("Scraping product details from Shopify API")
    exchange_rates = fetch_exchange_rates()
    scraped_products = scrape_all_products(
        category_product_pairs, categories_per_handle, exchange_rates
    )

    if not scraped_products:
        print("[ERROR] No products scraped. Exiting.")
        sys.exit(1)

    # --- Step 3: Fetch existing data & classify ---
    db = None
    new_products: List[Dict] = []
    updated_products: List[Dict] = []
    unchanged_urls: List[str] = []

    if args.skip_db:
        print_header("Skipping DB fetch — treating all as new (--skip-db)")
        for p in scraped_products:
            p["_needs_embedding"] = True
            p["_change_hash"] = compute_change_hash(p)
            p["_classification"] = "new"
        new_products = scraped_products
    else:
        print_header("Comparing scraped data against existing database")
        db = Database()
        existing_products = db.fetch_existing_products(config.SOURCE)

        new_products, updated_products, unchanged_urls = classify_products(
            scraped_products, existing_products, force=args.force
        )

        print(f"\n  Classification:")
        print(f"    • New:       {len(new_products)}")
        print(f"    • Updated:   {len(updated_products)}")
        print(f"    • Unchanged: {len(unchanged_urls)} (skipped)")

    products_needing_db = new_products + updated_products

    # --- Step 4: Generate embeddings (only for new + updated) ---
    if not args.skip_embeddings and products_needing_db:
        print_header(f"Generating embeddings for {len(products_needing_db)} products (new + updated)")
        embedder = SigLipEmbedder()
        products_needing_db = add_embeddings_to_products(
            products_needing_db, embedder,
            stagger_delay=config.EMBEDDING_STAGGER_DELAY,
        )
    elif args.skip_embeddings:
        print_header("Skipping embeddings (--skip-embeddings)")
    else:
        print_header("No products need embeddings — skipping")

    # --- Step 5: Batch upsert to Supabase ---
    upsert_results = {"success": 0, "failed": 0}

    if not args.skip_db and products_needing_db:
        print_header(f"Upserting {len(products_needing_db)} products to Supabase (batch size: {config.DB_BATCH_SIZE})")
        upsert_results = db.batch_upsert(products_needing_db)
    elif not args.skip_db:
        print_header("No products to upsert — all unchanged")
        upsert_results = {"success": 0, "failed": 0}
    else:
        print_header("Skipping database upsert (--skip-db)")

    # --- Step 6: Handle stale products ---
    stale_deleted = 0
    if not args.skip_db and not args.skip_stale:
        print_header("Checking for stale products")
        seen_urls = {p.get("product_url", "") for p in scraped_products if p.get("product_url")}
        state = load_state()
        previous_run_urls = set(state.get("last_run_urls", []))

        stale_urls = get_stale_urls(state, seen_urls)
        if stale_urls:
            print(f"[main] {len(stale_urls)} products have been missing for "
                  f"{config.STALE_THRESHOLD_RUNS}+ runs — deleting...")
            stale_deleted = db.delete_products_by_urls(config.SOURCE, stale_urls)
        else:
            print("[main] No stale products to delete.")

        # Save updated state for next run
        prev_strikes = state.get("strike_urls", {})
        save_state(seen_urls, previous_run_urls, prev_strikes)
        print(f"[main] State saved to {config.STATE_FILE}")
    elif args.skip_stale:
        print_header("Skipping stale product deletion (--skip-stale)")

    # --- Summary ---
    elapsed = time.time() - start_time
    print()
    print("=" * 72)
    print("  SCRAPE COMPLETE — RUN SUMMARY")
    print("=" * 72)
    print(f"  Duration:         {elapsed:.1f}s")
    print(f"  Total scraped:    {len(scraped_products)}")
    print(f"  New:              {len(new_products)}")
    print(f"  Updated:          {len(updated_products)}")
    print(f"  Unchanged (skip): {len(unchanged_urls)}")
    print(f"  Stale deleted:    {stale_deleted}")
    if not args.skip_db:
        print(f"  DB upserted:      {upsert_results['success']}")
        print(f"  DB failed:        {upsert_results['failed']}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
