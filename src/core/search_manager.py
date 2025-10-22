"""
SearchManager - Singleton class to manage search indices throughout application lifecycle.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import logging
import numpy as np

from src.core.config import get_settings
from src.core.embedding_manager import EmbeddingManager
from src.hybrid_search.my_bm25 import BM_25
from src.hybrid_search.fusion_layer import RakingFusion
from src.hybrid_search.my_bm25 import Document as BM25Doc
from src.hybrid_search.embedding_search import EmbeddingSearch
from src.llm_reranking.prompt_based.llm_reranker import LLMReranker

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document structure for search."""
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class SearchManager:
    """
    Singleton search manager that persists throughout application lifecycle.
    Manages BM25 and vector indices, provides unified search interface.
    """
    
    _instance: Optional['SearchManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'SearchManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(SearchManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize SearchManager (only once due to singleton)."""
        if self._initialized:
            return
        
        self.settings = get_settings()
        self.documents: List[Document] = []
        self.bm25_index: Optional[BM_25] = None
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.document_embeddings: Optional[Any] = None  # numpy array or faiss index
        
        self.indices_path = Path(self.settings.INDEX_DIR)
        
        self.bm25_index_path = self.indices_path / "arxiv_bm25.pkl"
        self.vector_index_path = self.indices_path / "arxiv_vector.faiss"
        self.metadata_path = self.indices_path / "arxiv_metadata.pkl"
        
        self.embedding_search: Optional[EmbeddingSearch] = None
        
        self._initialized = True
        
        if self._all_index_files_exist():
            try:
                if self.load_indices():
                    logger.info("Auto-loaded existing indices on startup")
                else:
                    logger.warning("Failed to auto-load indices")
            except Exception as e:
                logger.error(f"Error auto-loading indices: {e}")
        
        logger.info("SearchManager singleton created")
    
    def _all_index_files_exist(self) -> bool:
        """Check if all required index files exist (matching build_index.py output)."""
        return (
            self.bm25_index_path.exists() and
            self.vector_index_path.exists() and
            self.metadata_path.exists()
        )
    
    def is_initialized(self) -> bool:
        """Check if indices are loaded and ready."""
        return (
            self.bm25_index is not None and 
            self.embedding_manager is not None and 
            self.embedding_search is not None and
            self.embedding_search.index is not None and
            len(self.documents) > 0
        )
    
    def load_indices(self) -> bool:
        """
        Load existing indices created by build_index.py.
        Returns True if successfully loaded, False if need to build from scratch.
        """
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.documents = []
                for doc_id, paper_data in metadata.items():
                    self.documents.append(Document(
                        id=doc_id,
                        title=paper_data.get('title', 'Untitled'),
                        content=f"{paper_data.get('abstract', '')} {paper_data.get('category', '')}".strip(),
                        metadata=paper_data
                    ))
                logger.info(f"Loaded {len(self.documents)} documents from metadata")
            else:
                logger.warning("No metadata file found")
                return False
            
            if self.bm25_index_path.exists():
                with open(self.bm25_index_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                logger.info("Loaded BM25 index")
            else:
                logger.warning("No BM25 index found")
                return False
            
            self.embedding_manager = EmbeddingManager(
                backend="sentence-transformers",
                model_name="all-MiniLM-L6-v2"
            )
            
            if self.vector_index_path.exists():
                self.embedding_search = EmbeddingSearch(self.embedding_manager)
                import faiss
                self.embedding_search.index = faiss.read_index(str(self.vector_index_path))
                
                for i, doc in enumerate(self.documents):
                    self.embedding_search.faiss_idx_to_doc_id[i] = doc.id
                    self.embedding_search.doc_id_to_faiss_idx[doc.id] = i
                    
                logger.info("Loaded FAISS vector index")
            else:
                logger.warning("No vector index found")
                return False
            
            logger.info("All indices loaded successfully from build_index.py format")
            return True
            
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            return False
    
    def build_indices(self, documents: List[Document]) -> None:
        """
        Build BM25 and vector indices from documents.
        
        Args:
            documents: List of documents to index
        """
        logger.info(f"Building indices for {len(documents)} documents")
        
        self.indices_path.mkdir(parents=True, exist_ok=True)
        
        self.documents = documents
        
        logger.info("Building BM25 index...")
        doc_texts = [f"{doc.title} {doc.content}" for doc in documents]
        
        bm25_docs = [BM25Doc(text) for text in doc_texts]
        self.bm25_index = BM_25(bm25_docs)
        
        logger.info("Building vector embeddings...")
        self.embedding_manager = EmbeddingManager()
        embeddings_list = self.embedding_manager.get_batch_embedding(doc_texts)
        self.document_embeddings = self.embedding_manager.embeddings_to_numpy(embeddings_list)
        
        self.save_indices()
        logger.info("Indices built and saved successfully")
    
    def save_indices(self) -> None:
        """Save indices to disk for persistence."""
        try:
            docs_data = [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "metadata": doc.metadata or {}
                }
                for doc in self.documents
            ]
            with open(self.documents_path, 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)
            
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.document_embeddings, f)
            
            logger.info("Indices saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving indices: {e}")
            raise
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25 index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document_id, score) tuples
        """
        if not self.bm25_index:
            raise RuntimeError("BM25 index not initialized")
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [(self.documents[i].id, scores[i]) for i in top_indices]
        
        return results
    
    def search_vector(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using pre-built FAISS index from build_index.py.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document_id, score) tuples
        """
        if not self.embedding_search or self.embedding_search.index is None:
            raise RuntimeError("Vector index not initialized")
        
        # Use the pre-built FAISS index
        results = self.embedding_search.search(query, top_k)
        return results
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 10, 
        alpha: float = 0.5
    ) -> List[Tuple[str, float, Optional[str]]]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Balance between BM25 (alpha) and vector (1-alpha) search
            
        Returns:
            List of (document_id, score, explanation) tuples sorted by fused score
            Explanation is None for non-LLM reranked results
        """
        if not self.is_initialized():
            raise RuntimeError("Search indices not initialized")
        
        bm25_results = self.search_bm25(query, top_k * 2)
        
        vector_results = self.search_vector(query, top_k * 2)

        logger.info(f"BM25 results: {bm25_results[:5]}")
        logger.info(f"Vector results: {vector_results[:5]}")
        
        fusion = RakingFusion(
            bm_25_ranking=bm25_results,
            emb_rankings=vector_results,
            alpha=alpha,
            beta=1.0 - alpha
        )

        fused_results = [(doc_id, score, None) for doc_id, score in fusion.weighted_sum()]

        logger.info(f"Fused {len(fused_results)} results using hybrid search with alpha={alpha}")
        
        try:
            re_ranker = LLMReranker()
            
            docs_for_reranking = []
            text_to_doc_id = {}
            
            for doc_id, score, _ in fused_results:
                doc = self.get_document_by_id(doc_id)
                if doc:
                    doc_text = f"{doc.title}: {doc.content}"
                    docs_for_reranking.append((doc_text, score))
                    text_to_doc_id[doc_text] = doc_id
            
            if docs_for_reranking:
                logger.info(f"Starting LLM reranking for {len(docs_for_reranking)} documents")
                reranked_results = re_ranker.rerank(query, docs_for_reranking, top_k)
                logger.info(f"LLM reranking completed")
                
                # Convert reranked results back to (doc_id, score, explanation) format
                converted_results = []
                for reranked_text, new_score, explanation in reranked_results:
                    # Find the corresponding doc_id by matching title (more robust than full text)
                    found_match = False
                    for doc_id, score, _ in fused_results:
                        doc = self.get_document_by_id(doc_id)
                        if doc:
                            
                            if doc.title in reranked_text or reranked_text.startswith(doc.title):
                                converted_results.append((doc_id, new_score, explanation))
                                found_match = True
                                break
                    
                    if not found_match:
                        for doc_id, score, _ in fused_results:
                            doc = self.get_document_by_id(doc_id)
                            if doc:
                                full_doc_text = f"{doc.title}: {doc.content}"
                                if reranked_text in full_doc_text or full_doc_text.startswith(reranked_text[:50]):
                                    converted_results.append((doc_id, new_score, explanation))
                                    found_match = True
                                    break
                    
                    if not found_match:
                        logger.warning(f"Could not match reranked text: {reranked_text[:100]}...")
                
                if converted_results:
                    fused_results = [(doc_id, score, explanation) for doc_id, score, explanation in converted_results]
                    logger.info(f"Successfully converted {len(converted_results)} reranked results with explanations")
                else:
                    logger.warning("Failed to convert any reranked results, using original fusion results")
            
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}, using fusion results")

        fused_results.sort(key=lambda x: x[1], reverse=True)

        return fused_results[:top_k]
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def get_search_results(
        self, 
        query: str, 
        top_k: int = 10, 
        use_hybrid: bool = True,
        hybrid_alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get formatted search results for API responses.
        
        Args:
            query: Search query
            top_k: Number of results
            use_hybrid: Whether to use hybrid search
            hybrid_alpha: Hybrid search balance
            
        Returns:
            List of formatted search results
        """
        if not self.is_initialized():
            logger.error("Search indices not initialized")
            raise RuntimeError("Search indices not initialized. Run setup_indices.py first.")
        
        logger.info(f"Performing {'hybrid' if use_hybrid else 'BM25-only'} search for query: {query}")
        if use_hybrid:
            raw_results = self.hybrid_search(query, top_k, hybrid_alpha)
        else:
            raw_results = self.search_bm25(query, top_k)

        logger.info(f"Retrieved {len(raw_results)} raw results")
        
        formatted_results = []
        for result in raw_results:
            if len(result) == 3:
                doc_id, score, explanation = result
            else:
                doc_id, score = result
                explanation = None
                
            doc = self.get_document_by_id(doc_id)
            if doc:
                result_dict = {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "score": float(score),
                    "metadata": doc.metadata or {}
                }
                if explanation:
                    result_dict["llm_explanation"] = explanation
                formatted_results.append(result_dict)
        
        logger.info(f"Formatted {len(formatted_results)} results for output")
    
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search manager statistics."""
        return {
            "initialized": self.is_initialized(),
            "total_documents": len(self.documents),
            "has_bm25_index": self.bm25_index is not None,
            "has_vector_index": self.embedding_search is not None and self.embedding_search.index is not None,
            "indices_path": str(self.indices_path),
            "files_exist": {
                "bm25_index": self.bm25_index_path.exists(),
                "vector_index": self.vector_index_path.exists(),
                "metadata": self.metadata_path.exists(),
            }
        }


def get_search_manager() -> SearchManager:
    """Get the global SearchManager instance."""
    return SearchManager()