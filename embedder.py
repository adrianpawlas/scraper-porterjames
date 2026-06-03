"""Embedder module for generating image and text embeddings.

Uses google/siglip-base-patch16-384 from HuggingFace to produce 768-dimensional
embeddings for both images and text.
"""

import io
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

import config

# Suppress noisy SigLip model config warnings about bos/eos token ids
# These are known harmless warnings for the SigLip architecture
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*bos_token_id.*should be.*")
warnings.filterwarnings("ignore", message=".*eos_token_id.*should be.*")


def download_image(url: str, timeout: int = 15) -> Optional[Image.Image]:
    """Download an image from a URL and return as PIL Image."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"    [embedder] Failed to download {url[:80]}: {e}")
        return None


def download_images_parallel(urls: List[str], max_workers: int = 4) -> Dict[str, Image.Image]:
    """Download multiple images in parallel.

    Returns dict mapping URL → PIL Image for successful downloads.
    """
    results: Dict[str, Image.Image] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_image, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                img = future.result()
                if img is not None:
                    results[url] = img
            except Exception as e:
                print(f"    [embedder] Error downloading {url[:80]}: {e}")
    return results


class SigLipEmbedder:
    """Generates 768-dim embeddings using google/siglip-base-patch16-384."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[embedder] Loading model '{config.EMBEDDING_MODEL}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(config.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(config.EMBEDDING_MODEL).to(self.device)
        self.model.eval()
        print(f"[embedder] Model loaded. Embedding dimension: {config.EMBEDDING_DIM}")

    def _extract_embeds(self, outputs) -> torch.Tensor:
        """Extract pooled embedding tensor from model outputs.

        Handles both raw tensors (from get_image_features/get_text_features)
        and BaseModelOutputWithPooling objects.
        """
        if hasattr(outputs, "pooler_output"):
            embeds = outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            embeds = outputs
        else:
            raise TypeError(f"Unexpected output type: {type(outputs)}")
        # L2 normalize
        return embeds / embeds.norm(dim=-1, keepdim=True)

    def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
        """Generate embeddings for a list of PIL images.

        Processes images in batches to avoid OOM.
        Returns list of embedding lists (each 768-dim).
        """
        all_embeddings = []

        for i in range(0, len(images), config.EMBEDDING_BATCH_SIZE):
            batch = images[i : i + config.EMBEDDING_BATCH_SIZE]
            try:
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    embeds = self._extract_embeds(outputs)
                all_embeddings.extend(embeds.cpu().tolist())
            except Exception as e:
                print(f"    [embedder] Image batch embedding failed: {e}")
                for _ in batch:
                    all_embeddings.append([0.0] * config.EMBEDDING_DIM)

        return all_embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings.

        Returns list of embedding lists (each 768-dim).
        """
        all_embeddings = []

        for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE * 2):  # text is cheaper than images
            batch = texts[i : i + config.EMBEDDING_BATCH_SIZE * 2]
            try:
                inputs = self.processor(
                    text=batch,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model.get_text_features(**inputs)
                    embeds = self._extract_embeds(outputs)
                all_embeddings.extend(embeds.cpu().tolist())
            except Exception as e:
                print(f"    [embedder] Text batch embedding failed: {e}")
                for _ in batch:
                    all_embeddings.append([0.0] * config.EMBEDDING_DIM)

        return all_embeddings


def add_embeddings_to_products(
    products: List[Dict],
    embedder: SigLipEmbedder,
    stagger_delay: float = config.EMBEDDING_STAGGER_DELAY,
) -> List[Dict]:
    """Download product images and generate embeddings for each product.

    For each product:
    - Downloads the main image and generates image_embedding
    - Generates info_embedding from the text info

    Args:
        products: List of product dicts (will be mutated in place).
        embedder: Initialized SigLipEmbedder instance.
        stagger_delay: Seconds to wait between batches (rate limiting).

    Returns the products list with embeddings filled in.
    """
    products_to_embed = [p for p in products if p.get("_needs_embedding")]
    if not products_to_embed:
        print(f"[embedder] All {len(products)} products unchanged — skipping embeddings entirely.")
        # Even though we skip, ensure the unchanged products preserve existing embeddings
        return products

    skip_count = len(products) - len(products_to_embed)
    print(f"[embedder] Embedding {len(products_to_embed)} products ({skip_count} unchanged, skipped)")

    # --- Step 1: Generate info embeddings (text) — faster, do first ---
    print(f"\n[embedder] Generating text embeddings for {len(products_to_embed)} products...")
    info_texts = []
    for p in products_to_embed:
        text = p.get("_info_text", "")
        if not text:
            title = p.get("title", "")
            desc = p.get("description", "")
            price = p.get("price", "")
            sale = p.get("sale", "")
            category = p.get("category", "")
            tags = ", ".join(p.get("tags", [])) if p.get("tags") else ""
            size = p.get("size", "")
            gender = p.get("gender", "")
            brand = p.get("brand", "")
            text = (
                f"Title: {title}. "
                f"Description: {desc}. "
                f"Price: {price}. "
                f"Sale: {sale}. "
                f"Category: {category}. "
                f"Gender: {gender}. "
                f"Brand: {brand}. "
                f"Tags: {tags}. "
                f"Sizes: {size}. "
            )
        info_texts.append(text)

    info_embeddings = embedder.embed_texts(info_texts)
    for p, emb in zip(products_to_embed, info_embeddings):
        p["info_embedding"] = emb

    # --- Step 2: Download product images ---
    print(f"\n[embedder] Downloading product images for {len(products_to_embed)} products...")
    image_urls = []
    url_to_product_indices: Dict[str, List[int]] = {}

    for idx, p in enumerate(products):
        if not p.get("_needs_embedding"):
            continue
        url = p.get("image_url", "")
        if url:
            image_urls.append(url)
            if url not in url_to_product_indices:
                url_to_product_indices[url] = []
            url_to_product_indices[url].append(idx)

    unique_urls = list(set(image_urls))
    print(f"  [embedder] Downloading {len(unique_urls)} unique images...")

    downloaded = download_images_parallel(unique_urls)
    print(f"  [embedder] Successfully downloaded {len(downloaded)}/{len(unique_urls)} images")

    if stagger_delay > 0:
        time.sleep(stagger_delay)

    # --- Step 3: Generate image embeddings ---
    print(f"\n[embedder] Generating image embeddings...")
    image_list = []
    image_url_order = []

    for url in unique_urls:
        if url in downloaded:
            image_list.append(downloaded[url])
            image_url_order.append(url)

    if image_list:
        image_embeddings = embedder.embed_images(image_list)
        for url, emb in zip(image_url_order, image_embeddings):
            for idx in url_to_product_indices.get(url, []):
                products[idx]["image_embedding"] = emb
                if not products[idx].get("compressed_image_url"):
                    products[idx]["compressed_image_url"] = url

        if stagger_delay > 0:
            time.sleep(stagger_delay)

    # Log stats
    embedded_count = sum(1 for p in products if p.get("image_embedding"))
    text_embedded_count = sum(1 for p in products if p.get("info_embedding"))
    print(f"  [embedder] Image embeddings: {embedded_count}/{len(products)} products")
    print(f"  [embedder] Text embeddings: {text_embedded_count}/{len(products)} products")

    # Clean up internal fields before returning
    for p in products:
        p.pop("_needs_embedding", None)
        p.pop("_info_text", None)

    return products


