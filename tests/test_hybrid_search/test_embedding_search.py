"""
Tests for the EmbeddingSearch class.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.hybrid_search.embedding_search import EmbeddingSearch
from src.core.embedding_manager import EmbeddingManager


class TestEmbeddingSearchBasic:
    """Basic functionality tests for EmbeddingSearch."""
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock EmbeddingManager."""
        manager = Mock(spec=EmbeddingManager)
        manager.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
        manager.get_batch_embedding.return_value = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
        ]
        return manager
    
    @pytest.fixture
    def embedding_search(self, mock_embedding_manager):
        """Create an EmbeddingSearch instance."""
        return EmbeddingSearch(mock_embedding_manager)
    
    def test_initialization(self, embedding_search):
        """Test that EmbeddingSearch initializes correctly."""
        assert embedding_search.embedding_manager is not None
        assert embedding_search.index is None
    
    def test_load_embeddings(self, embedding_search, mock_embedding_manager):
        """Test loading embeddings for documents."""
        documents = ["doc1", "doc2", "doc3"]
        embeddings = embedding_search.load_embeddings(documents)
        
        assert len(embeddings) == 3
        assert len(embeddings[0]) == 4  # 4-dimensional embeddings
        mock_embedding_manager.get_batch_embedding.assert_called_once_with(documents)
    
    def test_dump_in_vector_store(self, embedding_search):
        """Test dumping embeddings into FAISS index."""
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
        ]
        documents = ["doc1", "doc2", "doc3"]
        
        embedding_search.dump_in_vector_store(embeddings, documents)
        
        assert embedding_search.index is not None
        assert embedding_search.index.ntotal == 3
    
    def test_search_without_index_raises_error(self, embedding_search):
        """Test that searching without an index raises an error."""
        with pytest.raises(ValueError, match="FAISS index is not initialized"):
            embedding_search.search("test query", top_k=5)


class TestEmbeddingSearchWithData:
    """Tests with actual FAISS index data."""
    
    @pytest.fixture
    def embedding_search_with_data(self):
        """Create an EmbeddingSearch instance with mock data."""
        manager = Mock(spec=EmbeddingManager)
        search = EmbeddingSearch(manager)
        
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],  # doc 0
            [0.0, 1.0, 0.0, 0.0],  # doc 1
            [0.0, 0.0, 1.0, 0.0],  # doc 2
            [0.0, 0.0, 0.0, 1.0],  # doc 3
            [0.5, 0.5, 0.0, 0.0],  # doc 4
        ]
        documents = ["doc0", "doc1", "doc2", "doc3", "doc4"]
        search.dump_in_vector_store(embeddings, documents)
        
        manager.get_embedding.return_value = [0.9, 0.1, 0.0, 0.0]
        
        return search, manager
    
    def test_search_returns_correct_format(self, embedding_search_with_data):
        """Test that search returns results in correct format."""
        search, _ = embedding_search_with_data
        results = search.search("test query", top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        assert all(isinstance(item[0], str) for item in results)  # doc_id is string
        assert all(isinstance(item[1], float) for item in results)  # score is float
    
    def test_search_returns_top_k_results(self, embedding_search_with_data):
        """Test that search returns correct number of results."""
        search, _ = embedding_search_with_data
        
        results_3 = search.search("test query", top_k=3)
        assert len(results_3) == 3
        
        results_5 = search.search("test query", top_k=5)
        assert len(results_5) == 5
    
    def test_search_scores_are_sorted(self, embedding_search_with_data):
        """Test that search results are sorted by score (descending)."""
        search, _ = embedding_search_with_data
        results = search.search("test query", top_k=5)
        
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "Scores should be in descending order"
    
    def test_search_finds_most_similar(self, embedding_search_with_data):
        """Test that search finds the most similar document."""
        search, manager = embedding_search_with_data
        
        # Query embedding is [0.9, 0.1, 0.0, 0.0], which is closest to doc 0 [1.0, 0.0, 0.0, 0.0]
        results = search.search("test query", top_k=1)
        
        top_doc_idx, top_score = results[0]
        assert top_doc_idx == "0", f"Expected doc 0 to be most similar, got doc {top_doc_idx}"
        assert top_score > 0.8, "Score should be high for very similar embedding"


class TestEmbeddingSearchIntegration:
    """Integration tests using real embeddings (still mocked, but more realistic)."""
    
    def test_full_workflow(self):
        """Test the complete workflow: load embeddings, add to index, search."""
        manager = Mock(spec=EmbeddingManager)
        
        documents = [
            "The cat sat on the mat",
            "The dog played in the park",
            "A bird flew in the sky",
        ]
        
        manager.get_batch_embedding.return_value = [
            [0.8, 0.2, 0.1],  # cat-related
            [0.2, 0.8, 0.1],  # dog-related
            [0.1, 0.1, 0.9],  # bird-related
        ]
        
        manager.get_embedding.return_value = [0.75, 0.15, 0.1]
        
        search = EmbeddingSearch(manager)
        
        embeddings = search.load_embeddings(documents)
        search.dump_in_vector_store(embeddings, documents)
        
        results = search.search("cat on mat", top_k=2)
        
        assert len(results) == 2
        top_doc_idx, _ = results[0]
        assert top_doc_idx == "0", "Cat document should be ranked first"
    
    def test_incremental_addition(self):
        """Test adding documents to the index incrementally."""
        manager = Mock(spec=EmbeddingManager)
        search = EmbeddingSearch(manager)
        
        # Add first batch
        embeddings_1 = [[1.0, 0.0], [0.0, 1.0]]
        documents_1 = ["doc0", "doc1"]
        search.dump_in_vector_store(embeddings_1, documents_1)
        assert search.index.ntotal == 2
        
        # Add second batch
        embeddings_2 = [[0.5, 0.5]]
        documents_2 = ["doc2"]
        search.dump_in_vector_store(embeddings_2, documents_2)
        assert search.index.ntotal == 3
    
    def test_large_batch(self):
        """Test handling a large batch of documents."""
        manager = Mock(spec=EmbeddingManager)
        search = EmbeddingSearch(manager)
        
        # Create 100 random embeddings
        np.random.seed(42)
        embeddings = np.random.rand(100, 128).tolist()
        documents = [f"doc{i}" for i in range(100)]
        
        search.dump_in_vector_store(embeddings, documents)
        
        assert search.index.ntotal == 100
        
        manager.get_embedding.return_value = embeddings[0]
        
        # Search should work
        results = search.search("test", top_k=10)
        assert len(results) == 10

class TestEmbeddingActualWorkflow:
    """
    Test the actual workflow of embedding generation, indexing, and searching.
    """
    emb_model = ""
    manager = EmbeddingManager(backend="sentence-transformers", model_name="all-MiniLM-L6-v2")
    search = EmbeddingSearch(manager)

    def test_workflow(self):
        """
        Test the complete workflow: load embeddings, add to index, search.
        """
        documents = [
            "Bed for cat, comfortable and warm",
            "Small toys for dogs to play with",
            "Bird cages and accessories",
        ]

        embeddings = self.manager.get_batch_embedding(documents)
        self.search.dump_in_vector_store(embeddings, documents)

        results = self.search.search("stuff for cat to sleep on", top_k=2)

        assert len(results) == 2
        top_doc_idx, _ = results[0]
        assert top_doc_idx == "0", "Cat document should be ranked first"


class TestEmbeddingSearchEdgeCases:
    """Edge case tests."""
    
    def test_search_with_top_k_larger_than_index(self):
        """Test searching with top_k larger than number of documents."""
        manager = Mock(spec=EmbeddingManager)
        search = EmbeddingSearch(manager)
        
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        documents = ["doc0", "doc1"]
        search.dump_in_vector_store(embeddings, documents)
        
        manager.get_embedding.return_value = [1.0, 0.0]
        
        # Request more results than available
        results = search.search("test", top_k=10)
        
        # Should return only 2 results (all available)
        assert len(results) == 2
    
    def test_single_document(self):
        """Test with only one document in the index."""
        manager = Mock(spec=EmbeddingManager)
        search = EmbeddingSearch(manager)
        
        embeddings = [[1.0, 0.0, 0.0]]
        documents = ["single_doc"]
        search.dump_in_vector_store(embeddings, documents)
        
        manager.get_embedding.return_value = [1.0, 0.0, 0.0]
        
        results = search.search("test", top_k=1)
        
        assert len(results) == 1
        assert results[0][0] == "0"
    
    def test_empty_query(self):
        """Test behavior with empty query."""
        manager = Mock(spec=EmbeddingManager)
        search = EmbeddingSearch(manager)
        
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        documents = ["doc0", "doc1"]
        search.dump_in_vector_store(embeddings, documents)
        
        manager.get_embedding.return_value = [0.0, 0.0]  # Empty embedding
        
        # Should still return results (though scores will be low)
        results = search.search("", top_k=2)
        assert len(results) == 2


@pytest.mark.slow
@pytest.mark.integration
class TestEmbeddingActualWorkflow:
    """
    Integration test with real sentence-transformers embeddings.
    This test is slower as it loads an actual embedding model.
    Run with: pytest -m integration
    Skip with: pytest -m "not slow"
    """
    
    @pytest.fixture(scope="class")
    def real_manager(self):
        """Create a real EmbeddingManager with sentence-transformers."""
        try:
            manager = EmbeddingManager(
                backend="sentence-transformers",
                model_name="all-MiniLM-L6-v2"
            )
            manager.preprocess_text = lambda text: text.strip().lower()
            return manager
        except ImportError:
            pytest.skip("sentence-transformers not installed")
    
    @pytest.fixture
    def real_search(self, real_manager):
        """Create a fresh EmbeddingSearch instance for each test."""
        return EmbeddingSearch(real_manager)

    def test_semantic_search_workflow(self, real_search, real_manager):
        """
        Test the complete workflow with real embeddings.
        Verifies that semantic similarity works correctly.
        """
        documents = [
            "Bed for cat, comfortable and warm",
            "Small toys for dogs to play with",
            "Bird cages and accessories",
        ]

        embeddings = real_manager.get_batch_embedding(documents)
        real_search.dump_in_vector_store(embeddings, documents)

        results = real_search.search("stuff for cat to sleep on", top_k=2)

        assert len(results) == 2, "Should return top 2 results"
        
        top_doc_idx, top_score = results[0]
        assert top_doc_idx == "0", f"Cat bed should be ranked first, got doc {top_doc_idx}"
        assert top_score > 0.3, f"Top result should have reasonable similarity score, got {top_score}"
        
        _, second_score = results[1]
        assert top_score > second_score, "Scores should be in descending order"
    
    def test_exact_match_high_score(self, real_search, real_manager):
        """Test that exact or near-exact matches get high scores."""
        documents = [
            "Python programming language",
            "Java programming language",
            "JavaScript web development",
        ]
        
        embeddings = real_manager.get_batch_embedding(documents)
        real_search.dump_in_vector_store(embeddings, documents)
        
        results = real_search.search("Python programming", top_k=1)
        
        top_doc_idx, top_score = results[0]
        assert top_doc_idx == "0", "Python doc should rank first"
        assert top_score > 0.6, f"Exact match should have high score, got {top_score}"
    
    def test_semantic_similarity_vs_keyword_match(self, real_search, real_manager):
        """Test that embeddings capture semantic meaning, not just keywords."""
        documents = [
            "The feline sat on the mat",  # Semantic match (cat = feline)
            "The category of products",   # Keyword match (contains 'cat')
            "A dog played in the yard",   # No match
        ]
        
        embeddings = real_manager.get_batch_embedding(documents)
        real_search.dump_in_vector_store(embeddings, documents)
        
        results = real_search.search("cat sitting", top_k=3)
        
        # Feline (semantic) should rank higher than category (keyword)
        top_doc_idx = results[0][0]
        assert top_doc_idx == "0", "Semantic similarity should beat keyword matching"
