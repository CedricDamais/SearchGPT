"""
Embedding Manager for SearchGPT.

Supports multiple embedding backends:
- Ollama (local)
- Sentence-Transformers (local)
- OpenAI API (remote)

All methods return list[float] for consistency and JSON serialization.
"""

from typing import List, Literal
import numpy as np
from src.core.config import settings
import spacy

EmbeddingBackend = Literal["ollama", "sentence-transformers", "openai"]


class EmbeddingManager:
    """
    Unified interface for different embedding models.
    
    All embeddings are returned as list[float] for consistency,
    regardless of the underlying library (numpy, torch, or native lists).
    """
    
    def __init__(
        self, 
        model_name: str = None,
        backend: EmbeddingBackend = "ollama"
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the embedding model
            backend: Which embedding service to use
        """
        self.model_name = model_name or settings.embedding_model
        self.backend = backend
        self._model = None
        self._initialize_backend()
    
    def _initialize_backend(self) -> None:
        """Initialize the selected backend."""
        if self.backend == "ollama":
            import ollama
            self._client = ollama
            
        elif self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            
        elif self.backend == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding as list[float] (e.g., [0.123, -0.456, ...])
            
        Example:
            >>> manager = EmbeddingManager()
            >>> embedding = manager.get_embedding("Hello world")
            >>> type(embedding)
            <class 'list'>
            >>> len(embedding)  # Dimension depends on model
            768
        """
        processed_text = self.preprocess_text(text=text)
        if self.backend == "ollama":
            response = self._client.embeddings(
                model=self.model_name,
                prompt=processed_text
            )
            return response['embedding']
        
        elif self.backend == "sentence-transformers":
            embedding = self._model.encode(processed_text)
            return embedding.tolist()
        
        elif self.backend == "openai":
            response = self._client.embeddings.create(
                model=self.model_name,
                input=processed_text
            )
            return response.data[0].embedding
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text, e.g., lowercased, stripped, lemmatized, etc.
            
        Example:
            >>> manager = EmbeddingManager()
            >>> preprocessed = manager.preprocess_text("  Hello World!  ")
            >>> preprocessed
            'hello world!'
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.strip().lower())
        preprocessed = " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])
        return preprocessed
    
    def get_batch_embedding(self, text_batch: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            text_batch: List of texts to embed
            
        Returns:
            List of embeddings, each as list[float]
            
        Example:
            >>> manager = EmbeddingManager()
            >>> embeddings = manager.get_batch_embedding(["text1", "text2"])
            >>> len(embeddings)
            2
            >>> type(embeddings[0])
            <class 'list'>
        """
        processed_batch = [self.preprocess_text(text) for text in text_batch]
        if self.backend == "ollama":
            # Process each text individually for Ollama
            return [self.get_embedding(text) for text in processed_batch]

        elif self.backend == "sentence-transformers":
            embeddings = self._model.encode(processed_batch)
            return embeddings.tolist()

        elif self.backend == "openai":
            response = self._client.embeddings.create(
                model=self.model_name,
                input=processed_batch
            )
            return [item.embedding for item in response.data]
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension (e.g., 768, 1536, etc.)
        """
        sample = self.get_embedding("test")
        return len(sample)
    
    def embeddings_to_numpy(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Convert list of embeddings to numpy array for FAISS indexing.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            numpy array of shape (n_embeddings, embedding_dim)
            
        Example:
            >>> embeddings = manager.get_batch_embedding(["text1", "text2"])
            >>> array = manager.embeddings_to_numpy(embeddings)
            >>> array.shape
            (2, 768)
        """
        return np.array(embeddings, dtype=np.float32)
    
    def update_embedding_model(self, new_model: str) -> None:
        """
        Update the embedding model.
        
        Args:
            new_model: Name of the new model to use
        """
        self.model_name = new_model
        self._initialize_backend()

