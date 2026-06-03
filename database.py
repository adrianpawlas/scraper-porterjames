"""Database module for interacting with Supabase.

Handles upserting product data into the 'products' table,
including vector embeddings.
"""

import json
from typing import Dict, List, Optional

from supabase import create_client, Client

import config


class Database:
    """Manages Supabase database operations."""

    def __init__(self):
        self.url: str = config.SUPABASE_URL
        self.key: str = config.SUPABASE_KEY
        self.table: str = config.SUPABASE_TABLE
        self.client: Client = create_client(self.url, self.key)
        print(f"[database] Connected to Supabase: {self.url}")

    def upsert_product(self, product: Dict) -> bool:
        """Upsert a single product into the database.

        Uses the unique constraint (source, product_url) for conflict resolution.
        """
        try:
            # Prepare the record
            record = {}

            # Map our fields to the database columns
            field_mapping = {
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

            for our_key, db_key in field_mapping.items():
                value = product.get(our_key)
                if value is not None and value != "" and value != []:
                    record[db_key] = value

            # Upsert using the unique constraint (source, product_url)
            resp = (
                self.client.table(self.table)
                .upsert(record, on_conflict="source, product_url")
                .execute()
            )

            return True
        except Exception as e:
            print(f"  [database] ERROR upserting product '{product.get('title', 'unknown')}': {e}")
            return False

    def upsert_products_batch(self, products: List[Dict], batch_size: int = 10) -> Dict[str, int]:
        """Upsert multiple products to the database.

        Args:
            products: List of product dicts
            batch_size: Number of products per batch insert

        Returns:
            Dict with 'success' and 'failed' counts
        """
        if not products:
            return {"success": 0, "failed": 0}

        success = 0
        failed = 0
        total = len(products)

        print(f"\n[database] Upserting {total} products to Supabase table '{self.table}'...")

        for i, product in enumerate(products, 1):
            title = product.get("title", "unknown")
            ok = self.upsert_product(product)
            if ok:
                success += 1
                if i % 5 == 0 or i == 1 or i == total:
                    print(f"  [{i}/{total}] ✓ {title}")
            else:
                failed += 1
                print(f"  [{i}/{total}] ✗ {title}")

        print(f"\n[database] Done! ✓ {success} succeeded, ✗ {failed} failed out of {total} total.")
        return {"success": success, "failed": failed}


def verify_connection() -> bool:
    """Verify that the Supabase connection is working."""
    try:
        db = Database()
        # Try to fetch a single row to verify connection
        resp = db.client.table(config.SUPABASE_TABLE).select("id").limit(1).execute()
        print(f"[database] Connection verified. Table '{config.SUPABASE_TABLE}' is accessible.")
        return True
    except Exception as e:
        print(f"[database] Connection failed: {e}")
        return False


if __name__ == "__main__":
    verify_connection()
