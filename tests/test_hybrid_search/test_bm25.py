"""
Tests for BM25 implementation.
Validates correctness and benchmarks heap vs sort performance.
"""

import pytest
import time
from src.hybrid_search.my_bm25 import Document, BM_25


class TestBM25Basic:
    """Basic functionality tests."""
    
    def test_document_creation(self):
        """Test Document class initialization."""
        doc = Document("machine learning is awesome")
        
        assert doc.text == "machine learning is awesome"
        assert doc.length == 4
        assert doc.term_freq == {
            "machine": 1,
            "learning": 1,
            "is": 1,
            "awesome": 1
        }
    
    def test_document_with_repeated_terms(self):
        """Test term frequency counting with repeated terms."""
        doc = Document("python python python is great")
        
        assert doc.term_freq["python"] == 3
        assert doc.term_freq["is"] == 1
        assert doc.term_freq["great"] == 1
    
    def test_bm25_initialization(self):
        """Test BM25 initialization."""
        docs = [
            Document("machine learning"),
            Document("deep learning"),
            Document("neural networks"),
        ]
        bm25 = BM_25(docs)
        
        assert len(bm25.documents) == 3
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75
        assert bm25.avg_doc_length == 2.0
    
    def test_bm25_scores(self):
        """Test BM25 scoring."""
        docs = [
            Document("machine learning is awesome"),
            Document("deep learning is powerful"),
            Document("natural language processing"),
        ]
        bm25 = BM_25(docs)
        
        scores = bm25.get_scores("learning")
        
        # Documents with "learning" should have higher scores
        assert scores[0] > 0  # "machine learning is awesome"
        assert scores[1] > 0  # "deep learning is powerful"
        assert scores[2] == 0  # "natural language processing"
    
    def test_get_top_n(self):
        """Test top-N retrieval."""
        docs = [
            Document("Python is a programming language"),
            Document("Java is also a programming language"),
            Document("Machine learning uses Python"),
            Document("Python programming is fun"),
        ]
        bm25 = BM_25(docs)
        
        results = bm25.get_top_n("Python programming", n=2)
        
        assert len(results) == 2
        # Results should be sorted by score
        assert results[0][1] >= results[1][1]
        # Top result should contain both terms
        assert "Python" in results[0][0] and "programming" in results[0][0]


class TestBM25Ranking:
    """Test ranking correctness."""
    
    def test_exact_match_scores_highest(self):
        """Documents with all query terms should score higher."""
        docs = [
            Document("machine learning"),
            Document("machine"),
            Document("learning"),
            Document("unrelated document"),
        ]
        bm25 = BM_25(docs)
        
        scores = bm25.get_scores("machine learning")
        
        # First doc has both terms, should score highest
        assert scores[0] > scores[1]  # "machine learning" > "machine"
        assert scores[0] > scores[2]  # "machine learning" > "learning"
        assert scores[0] > scores[3]  # "machine learning" > "unrelated"
    
    def test_term_frequency_affects_score(self):
        """More frequent terms should increase score."""
        docs = [
            Document("python python python programming"),
            Document("python programming"),
        ]
        bm25 = BM_25(docs)
        
        scores = bm25.get_scores("python")
        
        # First doc has "python" 3 times, should score higher
        assert scores[0] > scores[1]
    
    def test_idf_affects_score(self):
        """Rare terms should have higher IDF and affect score more."""
        docs = [
            Document("common word rare_term"),
            Document("common word common"),
            Document("common word common"),
        ]
        bm25 = BM_25(docs)
        
        # "rare_term" appears in 1/3 docs (rare)
        # "common" appears in 3/3 docs (common)
        rare_scores = bm25.get_scores("rare_term")
        common_scores = bm25.get_scores("common")
        
        # First doc should score higher for rare term
        assert rare_scores[0] > common_scores[0]


class TestBM25Optimization:
    """Test heap optimization vs full sort."""
    
    def test_heap_and_sort_same_results(self):
        """Both methods should return same top-N results."""
        docs = [
            Document(f"document {i} with some content about topic {i % 5}")
            for i in range(100)
        ]
        bm25 = BM_25(docs)
        
        query = "topic content"
        n = 10
        
        heap_results = bm25.get_top_n(query, n, use_heap=True)
        sort_results = bm25.get_top_n(query, n, use_heap=False)
        
        # Should return same documents (order and scores)
        assert len(heap_results) == len(sort_results) == n
        
        for (text1, score1), (text2, score2) in zip(heap_results, sort_results):
            assert text1 == text2
            assert abs(score1 - score2) < 1e-6  # Floating point comparison


class TestBM25EdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_query(self):
        """Empty query should return zero scores."""
        docs = [Document("test document")]
        bm25 = BM_25(docs)
        
        scores = bm25.get_scores("")
        assert scores[0] == 0.0
    
    def test_query_with_no_matches(self):
        """Query with no matching terms should return zero scores."""
        docs = [
            Document("machine learning"),
            Document("deep learning"),
        ]
        bm25 = BM_25(docs)
        
        scores = bm25.get_scores("quantum physics")
        assert all(score == 0.0 for score in scores)
    
    def test_single_document(self):
        """BM25 should work with single document."""
        docs = [Document("single document")]
        bm25 = BM_25(docs)
        
        scores = bm25.get_scores("document")
        assert len(scores) == 1
        assert scores[0] > 0
    
    def test_top_n_greater_than_corpus(self):
        """Requesting more results than documents should return all documents."""
        docs = [
            Document("doc 1"),
            Document("doc 2"),
        ]
        bm25 = BM_25(docs)
        
        results = bm25.get_top_n("doc", n=10)
        assert len(results) == 2


@pytest.fixture
def large_corpus():
    """Create a large corpus for benchmarking."""
    return [
        Document(f"document {i} about topic {i % 100} with content")
        for i in range(10000)
    ]


class TestBM25Performance:
    """Performance benchmarking tests."""
    
    def test_heap_vs_sort_small_n(self, large_corpus, benchmark=None):
        """Benchmark heap vs sort for small N (top-10)."""
        bm25 = BM_25(large_corpus)
        query = "topic content"
        n = 10
        
        bm25.get_top_n(query, n, use_heap=True)
        
        start = time.perf_counter()
        for _ in range(10):
            heap_results = bm25.get_top_n(query, n, use_heap=True)
        heap_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(10):
            sort_results = bm25.get_top_n(query, n, use_heap=False)
        sort_time = time.perf_counter() - start
        
        print(f"\nPerformance (10k docs, top-{n}):")
        print(f"  Heap method: {heap_time*1000:.2f}ms")
        print(f"  Sort method: {sort_time*1000:.2f}ms")
        print(f"  Speedup: {sort_time/heap_time:.2f}x")
        
        # For small N and large corpus, heap should be faster
        # (but allow some variance in timing)
        # This is more of an informational test than assertion
    
    def test_complexity_analysis(self):
        """Demonstrate O(M log N) vs O(M log M) complexity."""
        sizes = [100, 1000, 5000]
        n = 10
        
        print("\nComplexity Analysis (top-10 from N documents):")
        print(f"{'Docs':>6} {'Heap (ms)':>12} {'Sort (ms)':>12} {'Speedup':>10}")
        print("-" * 45)
        
        for size in sizes:
            docs = [
                Document(f"doc {i} with content {i % 10}")
                for i in range(size)
            ]
            bm25 = BM_25(docs)
            query = "content"
            
            # Benchmark heap
            start = time.perf_counter()
            for _ in range(5):
                bm25.get_top_n(query, n, use_heap=True)
            heap_time = (time.perf_counter() - start) * 1000 / 5
            
            # Benchmark sort
            start = time.perf_counter()
            for _ in range(5):
                bm25.get_top_n(query, n, use_heap=False)
            sort_time = (time.perf_counter() - start) * 1000 / 5
            
            speedup = sort_time / heap_time if heap_time > 0 else 1.0
            print(f"{size:>6} {heap_time:>11.2f} {sort_time:>11.2f} {speedup:>9.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
