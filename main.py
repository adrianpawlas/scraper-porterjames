#!/usr/bin/env python3
"""
Porter James Product Scraper

Full pipeline:
1. Crawl all categories → discover all product handles
2. Scrape each product's details via Shopify JSON API
3. Generate image & text embeddings using SigLIP (768-dim)
4. Upsert everything to Supabase

Usage:
    python main.py              # Run full pipeline
    python main.py --skip-embeddings   # Skip embedding generation
    python main.py --skip-db           # Skip database upsert
    python main.py --verify-db         # Only verify DB connection
    python main.py --help              # Show usage
"""

import argparse
import sys
import time
from datetime import datetime, timezone

from crawler import crawl_all_categories
from scraper import fetch_exchange_rates, scrape_all_products
from embedder import SigLipEmbedder, add_embeddings_to_products
from database import Database, verify_connection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Porter James Product Scraper - Full pipeline that scrapes all products, "
        "generates embeddings, and imports to Supabase."
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip image and text embedding generation",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database upsert (useful for testing scrape only)",
    )
    parser.add_argument(
        "--verify-db",
        action="store_true",
        help="Only verify the database connection and exit",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=0,
        help="Maximum number of products to scrape (0 = all, useful for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape and re-upsert even if products already exist",
    )
    return parser.parse_args()


def print_header(step: str):
    """Print a formatted step header."""
    width = 72
    print()
    print("=" * width)
    print(f"  STEP: {step}")
    print("=" * width)
    print()


def main():
    args = parse_args()
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print()
    print("=" * 72)
    print("  PORTER JAMES PRODUCT SCRAPER")
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
    print_header("Scraping product details")
    exchange_rates = fetch_exchange_rates()
    products = scrape_all_products(category_product_pairs, categories_per_handle, exchange_rates)

    if not products:
        print("[ERROR] No products scraped. Exiting.")
        sys.exit(1)

    # --- Step 3: Generate embeddings ---
    if not args.skip_embeddings:
        print_header("Generating embeddings (SigLIP 768-dim)")
        embedder = SigLipEmbedder()
        products = add_embeddings_to_products(products, embedder)
    else:
        print_header("Skipping embeddings (--skip-embeddings)")

    # --- Step 4: Upsert to Supabase ---
    if not args.skip_db:
        print_header("Upserting products to Supabase")
        db = Database()
        results = db.upsert_products_batch(products)
        print(f"  Results: {results['success']} succeeded, {results['failed']} failed")
    else:
        print_header("Skipping database upsert (--skip-db)")

    # --- Summary ---
    elapsed = time.time() - start_time
    print()
    print("=" * 72)
    print("  SCRAPING COMPLETE")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Products processed: {len(products)}")
    if not args.skip_db:
        print(f"  DB results: {results['success']} upserted, {results['failed']} failed")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
