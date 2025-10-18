import math
import heapq
from collections import Counter
from typing import List, Tuple

class Document:
    def __init__(self, text: str):
        self.text = text
        self.score = 0.0
        self.length = len(text.split())
        self.term_freq = {}

        for term in self.text.split():
            self.term_freq[term] = self.term_freq.get(term, 0) + 1

class BM_25:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with the provided documents.
        Preprocess documents and build necessary data structures.
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avg_doc_length = sum(doc.length for doc in documents) / len(documents) if documents else 0
        self.doc_freq = Counter()
        for doc in documents:
            self.doc_freq.update(doc.term_freq.keys())
    
    def set_scores(self, query: str) -> None:
        """
        Compute BM25 scores for the given query against all documents.
        Update each document's score attribute.
        
        Formula: BM25(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
        
        Where:
        - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
        - f(qi, D) = frequency of term qi in document D
        - |D| = length of document D
        - avgdl = average document length
        - k1, b = tuning parameters
        """
        query_terms = query.lower().split()
        N = len(self.documents)
        
        for doc in self.documents:
            score = 0.0
            
            for term in query_terms:
                if term not in doc.term_freq:
                    continue
                
                # Calculate IDF (Inverse Document Frequency)
                # n(qi) = number of documents containing term qi
                n_qi = self.doc_freq[term]
                idf = math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
                
                # Calculate term frequency component
                f_qi_D = doc.term_freq[term]
                
                # Normalize by document length
                normalized_length = doc.length / self.avg_doc_length
                
                # BM25 formula
                numerator = f_qi_D * (self.k1 + 1)
                denominator = f_qi_D + self.k1 * (1 - self.b + self.b * normalized_length)
                
                score += idf * (numerator / denominator)
            
            doc.score = score

    def get_scores(self, query: str) -> list[float]:
        """
        Compute BM25 scores for the given query against all documents.
        Return a list of scores corresponding to each document.
        
        Returns:
            List of BM25 scores, one per document in the same order as self.documents
        """
        self.set_scores(query)
        return [doc.score for doc in self.documents]

    def get_top_n(self, query: str, n: int, use_heap: bool = True) -> list[tuple[str, float]]:
        """
        Retrieve the top N documents most relevant to the query based on BM25 scores.
        
        Two implementations:
        1. Heap-based (O(M log N)): Efficient for small N, large M
        2. Full sort (O(M log M)): Simpler, good for small corpora
        
        Args:
            query: Search query string
            n: Number of top results to return
            use_heap: If True, use heap optimization; if False, use full sort
            
        Returns:
            List of tuples (document_text, score) sorted by relevance (highest first)
        """
        self.set_scores(query)
        
        if use_heap:
            # Heap-based approach: O(M log N) where M = total docs, N = top-k
            # More efficient when N << M (e.g., top-10 from 1M documents)
            top_docs = heapq.nlargest(
                n,
                self.documents,
                key=lambda doc: doc.score
            )
            return [(doc.text, doc.score) for doc in top_docs]
        else:
            sorted_docs = sorted(
                self.documents,
                key=lambda doc: doc.score,
                reverse=True
            )
            return [(doc.text, doc.score) for doc in sorted_docs[:n]]