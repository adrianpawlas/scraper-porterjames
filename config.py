"""Configuration constants for the Porter James scraper."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Brand / Source ---
BRAND = "Porter James"
SOURCE = "scraper-porterjames"
GENDER = "man"
SECOND_HAND = False
COUNTRY = "NZ"

# --- Store URLs ---
BASE_URL = "https://porterjames.com"
COLLECTIONS_URL = f"{BASE_URL}/collections.json"
PRODUCTS_JSON_URL = f"{BASE_URL}/products.json"

# Category handles to scrape (from the provided URLs)
CATEGORY_HANDLES = [
    "pants",
    "all-shorts",
    "denim",
    "shirting",
    "knitwear",
    "outerwear",
    "blazers",
    "tees",
    "headwear",
    "leather-goods",
    "socks",
]

# Map handles to display categories
CATEGORY_MAP = {
    "pants": "Pants",
    "all-shorts": "Shorts",
    "denim": "Denim",
    "shirting": "Shirting",
    "knitwear": "Knitwear",
    "outerwear": "Outerwear",
    "blazers": "Blazers",
    "tees": "Tees",
    "headwear": "Headwear",
    "leather-goods": "Leather Goods",
    "socks": "Socks",
}

# --- Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)
SUPABASE_TABLE = "products"

# --- Embedding Model ---
EMBEDDING_MODEL = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768
BATCH_SIZE = 8  # images per batch for embedding

# --- Scraping ---
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
SHOPIFY_PAGE_LIMIT = 250  # max products per page from Shopify API

# --- Currency Conversion ---
# Since the store uses CZK and user wants EUR/USD primarily.
# We'll fetch live rates from frankfurter.app (free, no API key needed).
EXCHANGE_RATE_API = "https://api.frankfurter.app/latest?from=NZD&to=EUR,USD"

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
