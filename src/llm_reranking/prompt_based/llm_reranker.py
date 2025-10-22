from src.core.ollama_client import OllamaClient
from src.core.logging import logger
from ollama import chat, ChatResponse
from typing import List, Tuple
from pydantic import BaseModel

"""
LLM Reranker using Ollama models for prompt-based reranking.
On another module we will have another fine tuned LLM Reranker. ( Cross-Encoder based)
"""

class ReRankedDocument(BaseModel):
    text: str
    new_score: float
    explanation: str

class ReRankedDocuments(BaseModel):
    documents: List[ReRankedDocument]


class LLMReranker: 

    def __init__(self, model_name: str = "gemma3:1b"):
        """
        Initialize the LLM Reranker with the specified Ollama model.
        
        Args:
            model_name: Name of the Ollama model to use for reranking
        """
        self.model_name = model_name
        self.client = OllamaClient(model_name=model_name)
        self.model_ranking : List[Tuple[str, float]] = []
    
    def structured_output(self, ):
        """
        Tool for the model to output structured data.
        """
        self.model_ranking = []
        
    def rerank(self, query: str, documents: list[tuple[str, float]], top_k : int) -> list[tuple[str, float, str]]:
        """
        Rerank documents based on the query using the Ollama model.
        
        Args:
            query: The search query
            documents: List of tuples (document text, original score)
            top_k: Number of top documents to return after reranking
            
        Returns:
            List of tuples (document text, reranked score, explanation)
        """
        prompt = self._build_prompt(query, documents)
        logger.info(f"LLM Reranker prompt: {prompt}")
        response = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=ReRankedDocuments.model_json_schema(),
            options={
                "temperature": 0.1,
                "timeout": 60
            }
        )
        logger.info(f"LLM Reranker response: {response}")
        documents = ReRankedDocuments.model_validate_json(response.message.content)
        documents_list = [(doc.text, doc.new_score, doc.explanation) for doc in documents.documents]
        logger.info(f"Reranked documents with explanations: {documents_list}")
        return documents_list

    def _build_prompt(self, query: str, documents: list[tuple[str, float]]) -> str:
        """
        Build the prompt for the LLM based on the query and documents.
        
        Args:
            query: The search query
            documents: List of tuples (document text, original score)
            
        Returns:
            Formatted prompt string
        """
        doc_texts = "\n\n".join([f"Document {i+1}:\n{doc[0]}" for i, doc in enumerate(documents)])
        prompt = (
            f"You are a helpful assistant that reranks search results based on relevance to the query.\n"
            f"Query: {query}\n\n"
            f"Here are the documents:\n{doc_texts}\n\n"
            f"Please provide a reranked list of documents based on their relevance to the query."
        )
        return prompt
