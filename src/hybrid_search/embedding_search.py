from src.core.embedding_manager import EmbeddingManager
import faiss
import numpy as np
import pickle
from pathlib import Path
from src.core.logging import logger


class EmbeddingSearch:
    """
    Vector search using embeddings and FAISS.
    
    Maintains a mapping between FAISS indices and document IDs for:
    1. Returning actual documents to users
    2. Aligning with BM25 results during fusion
    """
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.index = None  # FAISS index will be initialized later
        self.doc_id_to_faiss_idx = {}  # Map: doc_id -> FAISS index position
        self.faiss_idx_to_doc_id = {}  # Map: FAISS index position -> doc_id
        logger.info("Initialized EmbeddingSearch with FAISS index ...")
        
    
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Perform embedding-based search.
        """
        query_embedding = self.embedding_manager.get_embedding(query)
        logger.info(f"Generated embedding for query: {query}")
        results = self.load_embeddings_from_vector_store(query_embedding, top_k)
        logger.info(f"Search completed for query: {query}")
        return results

    def load_embeddings(self, documents: list[str]) -> list[list[float]]:
        """
        Load embeddings for a list of documents.
        """
        logger.info(f"Loading embeddings for {len(documents)} documents...")
        embeddings = self.embedding_manager.get_batch_embedding(documents)
        logger.info("Embeddings loaded successfully.")
        return embeddings
    
    def dump_in_vector_store(self, embeddings: list[list[float]], documents: list[str], doc_ids: list[str] = None) -> None:
        """
        Add embeddings to the vector store and maintain document mapping.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of document texts (for reference)
            doc_ids: Optional list of document IDs. If None, uses sequential integers
            
        Example:
            >>> search.dump_in_vector_store(
            ...     embeddings=[[0.1, 0.2], [0.3, 0.4]],
            ...     documents=["doc1 text", "doc2 text"],
            ...     doc_ids=["prod_001", "prod_002"]
            ... )
        """
        if self.index is None:
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatIP(dimension)
        
        # Generate doc_ids if not provided
        if doc_ids is None:
            start_idx = len(self.faiss_idx_to_doc_id)
            doc_ids = [str(i) for i in range(start_idx, start_idx + len(documents))]
        
        vectors = np.array(embeddings).astype('float32')
        start_faiss_idx = self.index.ntotal
        self.index.add(vectors)
        
        # Update mappings
        for i, doc_id in enumerate(doc_ids):
            faiss_idx = start_faiss_idx + i
            self.doc_id_to_faiss_idx[doc_id] = faiss_idx
            self.faiss_idx_to_doc_id[faiss_idx] = doc_id
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")
    
    def save_index(self, index_path: str, metadata_path: str = None) -> None:
        """
        Save FAISS index and document mappings to disk.
        
        Args:
            index_path: Path where to save the FAISS index file
            metadata_path: Optional path for metadata. If None, derives from index_path
            
        Example:
            >>> search.save_index('data/indices/products.faiss')
            # Also saves: data/indices/products_metadata.pkl
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first using dump_in_vector_store().")
        
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path} ({self.index.ntotal} vectors)")
        
        # Save metadata (document mappings)
        if metadata_path is None:
            metadata_path = str(Path(index_path).with_suffix('.pkl'))
        
        metadata = {
            'doc_id_to_faiss_idx': self.doc_id_to_faiss_idx,
            'faiss_idx_to_doc_id': self.faiss_idx_to_doc_id,
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str = None) -> None:
        """
        Load FAISS index and document mappings from disk.
        
        Args:
            index_path: Path to the saved FAISS index file
            metadata_path: Optional path for metadata. If None, derives from index_path
            
        Example:
            >>> search.load_index('data/indices/products.faiss')
            # Also loads: data/indices/products_metadata.pkl
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path} ({self.index.ntotal} vectors)")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = str(Path(index_path).with_suffix('.pkl'))
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.doc_id_to_faiss_idx = metadata['doc_id_to_faiss_idx']
        self.faiss_idx_to_doc_id = metadata['faiss_idx_to_doc_id']
        
        logger.info(f"Loaded metadata from {metadata_path} ({len(self.faiss_idx_to_doc_id)} documents)")
        
    def load_embeddings_from_vector_store(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """
        Search the vector store and return doc_ids with scores.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
            
        Example:
            >>> results = search.load_embeddings_from_vector_store([0.1, 0.2, ...], top_k=5)
            >>> results
            [("prod_001", 0.95), ("prod_042", 0.87), ...]
        """
        if self.index is None:
            raise ValueError("FAISS index is not initialized.")
        
        logger.info("Searching in FAISS index...")
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        logger.info(f"Search completed in FAISS index")

        results = []
        for dist, faiss_idx in zip(distances[0], indices[0]):
            if faiss_idx != -1:
                doc_id = self.faiss_idx_to_doc_id.get(int(faiss_idx), str(faiss_idx))
                results.append((doc_id, float(dist)))
        
        return results