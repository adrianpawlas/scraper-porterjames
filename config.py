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
EMBEDDING_BATCH_SIZE = 8  # images per batch for embedding
EMBEDDING_STAGGER_DELAY = 0.5  # seconds between embedding calls

# --- Scraping ---
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
SHOPIFY_PAGE_LIMIT = 250  # max products per page from Shopify API

# Rate limiting — Shopify can be aggressive about 429s
RATE_LIMIT_DELAY = 0.6  # seconds between individual product API requests
RATE_LIMIT_429_DELAY = 5.0  # base seconds to wait on 429 before retrying

# --- Database ---
DB_BATCH_SIZE = 50  # products per batch insert
DB_UPSERT_RETRIES = 3
STALE_THRESHOLD_RUNS = 2  # number of consecutive runs missed before deletion

# --- Currency Conversion ---
EXCHANGE_RATE_API = "https://api.frankfurter.app/latest?from=NZD&to=EUR,USD"

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
STATE_FILE = os.path.join(DATA_DIR, "scraper_state.json")
FAILED_PRODUCTS_LOG = os.path.join(DATA_DIR, "failed_products.log")

os.makedirs(IMAGES_DIR, exist_ok=True)
