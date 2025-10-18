from src.core.embedding_manager import EmbeddingManager


class EmbeddingLoader:
    """
    Load and manage embeddings for documents.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
    
    def load_embeddings(self, documents: list[str]) -> list[list[float]]:
        """
        Load embeddings for a list of documents.
        """
        embeddings = self.embedding_manager.get_batch_embedding(documents)
        return embeddings
    
    def dump_in_vector_store(self, embeddings: list[list[float]], documents: list[str]) -> None:
        """
        Drop embeddings into the vector store.
        """
        pass

    def load_embeddings_from_vector_store(self, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """
        Load top_k embeddings from vector store based on query embedding.
        """
        pass

class EmbeddingSearch:
    """
    Hybrid search using embeddings.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
    
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Perform embedding-based search.
        """
        query_embedding = self.embedding_manager.get_embedding(query)
        # Search logic to find top_k similar documents based on query_embedding
        pass