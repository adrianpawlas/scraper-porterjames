"""Database module for interacting with Supabase.

Provides:
- Batch upserts (50 per batch) with retries
- Change detection via content hashing
- Fetching existing products for comparison
- Stale product deletion
- Run summary reporting
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Set, Tuple

from supabase import create_client, Client

import config

# Fields that determine whether a product has "changed"
# (i.e. if any of these differ between the scraped and stored version, it's an update)
CHANGE_SENSITIVE_FIELDS = [
    "title",
    "description",
    "price",
    "sale",
    "image_url",
    "additional_images",
    "category",
    "size",
    "tags",
    "metadata",
]


def compute_change_hash(product: Dict) -> str:
    """Compute a hash of the change-sensitive fields of a product.

    Two products with the same hash are considered identical — no update needed.
    """
    relevant = {}
    for field in CHANGE_SENSITIVE_FIELDS:
        val = product.get(field)
        # Normalize for comparison
        if isinstance(val, list):
            val = json.dumps(val, sort_keys=True)
        elif val is None:
            val = ""
        relevant[field] = val

    raw = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Mapping from internal field names to DB column names
FIELD_MAPPING = {
    "id": "id",
    "source": "source",
    "product_url": "product_url",
    "affiliate_url": "affiliate_url",
    "image_url": "image_url",
    "brand": "brand",
    "title": "title",
    "description": "description",
    "category": "category",
    "gender": "gender",
    "search_tsv": "search_tsv",
    "created_at": "created_at",
    "metadata": "metadata",
    "size": "size",
    "second_hand": "second_hand",
    "image_embedding": "image_embedding",
    "country": "country",
    "compressed_image_url": "compressed_image_url",
    "tags": "tags",
    "search_vector": "search_vector",
    "title_tsv": "title_tsv",
    "brand_tsv": "brand_tsv",
    "description_tsv": "description_tsv",
    "other": "other",
    "price": "price",
    "sale": "sale",
    "additional_images": "additional_images",
    "info_embedding": "info_embedding",
}


def _record_from_product(product: Dict) -> Dict:
    """Build a DB record from a product dict, filtering out None/empty/internal fields."""
    record = {}
    for our_key, db_key in FIELD_MAPPING.items():
        value = product.get(our_key)
        if value is not None and value != "" and value != []:
            # If the value is a list like tags, keep it; only skip empty lists
            if isinstance(value, list) and len(value) == 0:
                continue
            record[db_key] = value
        elif our_key == "image_url" and value is None:
            # Skip image_url if None — let the NOT NULL constraint be
            # handled by not including it; the product likely has one anyway.
            pass
    return record


class Database:
    """Manages Supabase database operations with smart upsert logic."""

    def __init__(self):
        self.url: str = config.SUPABASE_URL
        self.key: str = config.SUPABASE_KEY
        self.table: str = config.SUPABASE_TABLE
        self.client: Client = create_client(self.url, self.key)
        print(f"[database] Connected to Supabase: {self.url}")

    # ------------------------------------------------------------------
    # Fetch existing data
    # ------------------------------------------------------------------

    def fetch_existing_products(self, source: str) -> Dict[str, Dict]:
        """Fetch all existing products for a given source.

        Returns a dict mapping product_url → full product dict (including
        change_hash if previously stored in metadata, or computed).
        """
        print(f"[database] Fetching existing products for source '{source}'...")
        all_products: Dict[str, Dict] = {}

        page = 0
        page_size = 1000

        while True:
            resp = (
                self.client.table(self.table)
                .select("*")
                .eq("source", source)
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            rows = resp.data if resp.data else []
            if not rows:
                break

            for row in rows:
                product_url = row.get("product_url", "")
                if product_url:
                    # Add the stored change_hash if we can derive it from metadata
                    metadata_raw = row.get("metadata")
                    if isinstance(metadata_raw, str):
                        try:
                            metadata = json.loads(metadata_raw)
                            row["_stored_hash"] = metadata.get("_change_hash")
                        except (json.JSONDecodeError, TypeError):
                            pass
                    all_products[product_url] = dict(row)

            if len(rows) < page_size:
                break
            page += 1

        print(f"[database] Found {len(all_products)} existing products.")
        return all_products

    def fetch_existing_urls(self, source: str) -> Set[str]:
        """Quickly fetch just the product URLs for a source."""
        urls: Set[str] = set()
        page = 0
        page_size = 1000

        while True:
            resp = (
                self.client.table(self.table)
                .select("product_url")
                .eq("source", source)
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            rows = resp.data if resp.data else []
            if not rows:
                break
            for row in rows:
                url = row.get("product_url")
                if url:
                    urls.add(url)
            if len(rows) < page_size:
                break
            page += 1

        return urls

    # ------------------------------------------------------------------
    # Batch upsert with retries
    # ------------------------------------------------------------------

    def batch_upsert(self, products: List[Dict], batch_size: int = None) -> Dict[str, int]:
        """Upsert products in batches with retry on failure.

        Args:
            products: List of product dicts to upsert.
            batch_size: Max products per batch (default: config.DB_BATCH_SIZE).

        Returns:
            Dict with 'success' and 'failed' counts.
        """
        if not products:
            return {"success": 0, "failed": 0}

        batch_size = batch_size or config.DB_BATCH_SIZE
        total = len(products)
        success = 0
        failed = 0

        batches = [products[i : i + batch_size] for i in range(0, total, batch_size)]
        print(f"\n[database] Upserting {total} products in {len(batches)} batch(es) of up to {batch_size}...")

        for batch_idx, batch in enumerate(batches, 1):
            records = []
            for p in batch:
                rec = _record_from_product(p)

                # Embed the change hash in metadata for future comparison
                change_hash = p.get("_change_hash")
                if change_hash and "metadata" in rec:
                    try:
                        meta = json.loads(rec["metadata"]) if isinstance(rec["metadata"], str) else rec["metadata"]
                        if isinstance(meta, dict):
                            meta["_change_hash"] = change_hash
                            rec["metadata"] = json.dumps(meta)
                    except (json.JSONDecodeError, TypeError):
                        pass

                records.append(rec)

            # Retry loop for this batch
            last_error = None
            for attempt in range(1, config.DB_UPSERT_RETRIES + 1):
                try:
                    self.client.table(self.table).upsert(
                        records,
                        on_conflict="source, product_url",
                    ).execute()
                    success += len(batch)
                    titles = [p.get("title", "?") for p in batch]
                    print(f"  [batch {batch_idx}/{len(batches)}] ✓ {len(batch)} products ({titles[0]}...{titles[-1]})")
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < config.DB_UPSERT_RETRIES:
                        wait = 2 ** attempt
                        print(f"  [batch {batch_idx}/{len(batches)}] ⚠ retry {attempt}/{config.DB_UPSERT_RETRIES} after {wait}s: {e}")
                        time.sleep(wait)

            if last_error:
                failed += len(batch)
                print(f"  [batch {batch_idx}/{len(batches)}] ✗ FAILED after {config.DB_UPSERT_RETRIES} retries: {last_error}")
                # Log failed products
                self._log_failed_products(batch, str(last_error))

            # Small delay between batches to be polite
            if batch_idx < len(batches):
                time.sleep(0.3)

        print(f"\n[database] Batch upsert complete: ✓ {success} succeeded, ✗ {failed} failed")
        return {"success": success, "failed": failed}

    # ------------------------------------------------------------------
    # Stale product deletion
    # ------------------------------------------------------------------

    def delete_products_by_urls(self, source: str, urls: Set[str]) -> int:
        """Delete products matching given URLs for this source.

        Returns the number of deleted products.
        """
        if not urls:
            print("[database] No stale products to delete.")
            return 0

        url_list = list(urls)
        deleted = 0

        # Delete in batches (Supabase `in` filter supports lists)
        batch_size = 50
        for i in range(0, len(url_list), batch_size):
            batch = url_list[i : i + batch_size]
            try:
                resp = (
                    self.client.table(self.table)
                    .delete()
                    .eq("source", source)
                    .in_("product_url", batch)
                    .execute()
                )
                n = len(resp.data) if resp.data else 0
                deleted += n
                print(f"  [database] Deleted {n} stale products (batch {i // batch_size + 1})")
            except Exception as e:
                print(f"  [database] Failed to delete batch: {e}")

        print(f"[database] Total stale products deleted: {deleted}")
        return deleted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_failed_products(products: List[Dict], error: str):
        """Log failed product info to a local file."""
        try:
            with open(config.FAILED_PRODUCTS_LOG, "a") as f:
                f.write(f"\n--- Failed batch at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ---\n")
                f.write(f"Error: {error}\n")
                for p in products:
                    f.write(f"  {p.get('product_url', '?')} | {p.get('title', '?')}\n")
        except Exception:
            pass  # best effort


def verify_connection() -> bool:
    """Verify that the Supabase connection is working."""
    try:
        db = Database()
        resp = db.client.table(config.SUPABASE_TABLE).select("id").limit(1).execute()
        print(f"[database] Connection verified. Table '{config.SUPABASE_TABLE}' is accessible.")
        return True
    except Exception as e:
        print(f"[database] Connection failed: {e}")
        return False


if __name__ == "__main__":
    verify_connection()
