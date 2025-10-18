import ollama
from src.core.config import settings

"""
Ollama client for LLM interactions. This will be used for the prompt based
re-ranking of documents. it will be used in the LLM re-ranking module.
"""


class OllamaClient:
    
    def __init__(self, model_name = None, embedding_model = None):
        self.model = model_name or settings.RERANKING_MODEL
        self.embedding = embedding_model or settings.EMBEDDING_MODEL
    
    def llm_generate(self, query, full_message : bool = False, token_limit : int = 0):
        """
        Generate response from Ollama LLM.
        """
        
        if token_limit == 0:
            response = ollama.generate(
                model=self.model,
                prompt=query
            )
        else:
            response = ollama.generate(
                model=self.model,
                prompt=query,
            )
        
        if full_message:
            return response
        else:
            return response["message"]
    
    def get_embedding(self, text : str ) -> str:
        pass
    
    def get_batch_embedding(self, text_batch : list[str]) -> list[str]:
        pass

    def rerank(self, query : str, documents : list[str]) -> list[tuple[str, float]]:
        """
        Rerank documents based on relevance to the query using LLM.
        Prompt based approach.
        """
        pass