"""
Script to build FAISS index from product dataset.
Run this once after loading/updating your dataset.

Usage:
    python scripts/build_index.py --dataset data/datasets/products.json --output data/indices/
"""

import argparse
import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embedding_manager import EmbeddingManager
from src.hybrid_search.embedding_search import EmbeddingSearch
from src.hybrid_search.my_bm25 import BM_25, Document
from src.core.logging import logger
from src.core.config import settings
import faiss


def build_vector_index(products: list[dict], output_dir: Path) -> None:
    """Build and save FAISS vector index."""
    logger.info(f"Building vector index for {len(products)} products...")

    documents = []
    for product in products:
        text = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')}"
        documents.append(text.strip())
    
    manager = EmbeddingManager(
        backend="sentence-transformers",
        model_name="all-MiniLM-L6-v2"
    )
    manager.preprocess_text = lambda text: text.strip().lower()
    
    print("Generating embeddings...")
    embeddings = manager.get_batch_embedding(documents)
    
    logger.info("Generating embeddings completed.")
    search = EmbeddingSearch(manager)
    search.dump_in_vector_store(embeddings, documents)
    logger.info("FAISS index built successfully.")
    
    index_path = settings.INDEX_DIR / "products_vector.faiss"
    faiss.write_index(search.index, str(index_path))
    logger.info(f"Vector index saved to {index_path}")

    metadata_path = output_dir / "products_metadata.pkl"
    metadata = {str(i): products[i] for i in range(len(products))}
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Metadata saved to {metadata_path}")

    return search


def build_bm25_index(products: list[dict], output_dir: Path) -> None:
    """Build and save BM25 index."""
    print(f"Building BM25 index for {len(products)} products...")
    
    documents = []
    for i, product in enumerate(products):
        text = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')}"
        doc = Document(id=str(i), text=text.strip())
        documents.append(doc)

    bm25 = BM_25(documents)
    
    bm25_path = output_dir / "products_bm25.pkl"
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved to {bm25_path}")


def main():
    parser = argparse.ArgumentParser(description="Build search indices from product dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to product dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=settings.INDEX_DIR,
        help="Output directory for indices"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["vector", "bm25", "both"],
        default="both",
        help="Type of index to build"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        products = json.load(f)
    
    print(f"Loaded {len(products)} products")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.index_type in ["vector", "both"]:
        build_vector_index(products, output_dir)
    
    if args.index_type in ["bm25", "both"]:
        build_bm25_index(products, output_dir)

    logger.info("Index building complete!")
    logger.info(f"Indices saved to: {output_dir}")


if __name__ == "__main__":
    main()
