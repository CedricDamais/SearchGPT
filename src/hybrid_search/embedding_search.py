from src.core.embedding_manager import EmbeddingManager
import faiss
import numpy as np
from src.core.logging import logger
class EmbeddingSearch:
    """
    Hybrid search using embeddings.
    """
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.index = None  # FAISS index will be initialized later
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
    
    def dump_in_vector_store(self, embeddings: list[list[float]], documents: list[str]) -> None:
        """
        Drop embeddings into the vector store.
        """
        if self.index is None:
            dimension = len(embeddings[0])
            self.index = faiss.IndexFlatIP(dimension)

        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
    
    def save_index(self, index_path: str) -> None:
        """
        Save FAISS index to disk.
        
        Args:
            index_path: Path where to save the index file
            
        Example:
            >>> search.save_index('data/indices/products.faiss')
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first using dump_in_vector_store().")
        
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path} ({self.index.ntotal} vectors)")
    
    def load_index(self, index_path: str) -> None:
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to the saved index file
            
        Example:
            >>> search.load_index('data/indices/products.faiss')
        """
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path} ({self.index.ntotal} vectors)")
        
    def load_embeddings_from_vector_store(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """
        Load top_k embeddings from vector store based on query embedding.
        """
        if self.index is None:
            raise ValueError("FAISS index is not initialized.")
        
        logger.info("Searching in FAISS index...")
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        logger.info(f"Search completed in FAISS index for query: {query_embedding}")

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Filter out invalid indices (FAISS returns -1 when top_k > num_docs)
            if idx != -1:
                results.append((str(idx), float(dist)))
        
        return results