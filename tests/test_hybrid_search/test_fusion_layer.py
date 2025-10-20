import pytest
from src.hybrid_search.fusion_layer import RakingFusion


class TestRakingFusion:
    """Test suite for the RakingFusion class."""
    
    def setup_method(self):
        """Setup test data for each test method."""
        self.bm25_ranking = [
            ("doc1", 0.8),
            ("doc2", 0.6),
            ("doc3", 0.4),
            ("doc4", 0.2)
        ]

        self.emb_rankings = [
            ("doc1", 0.9),
            ("doc2", 0.3),
            ("doc3", 0.7),
            ("doc4", 0.5)
        ]

        self.alpha = 0.4
        self.beta = 0.6
        
        self.fusion = RakingFusion(
            bm_25_ranking=self.bm25_ranking,
            emb_rankings=self.emb_rankings,
            alpha=self.alpha,
            beta=self.beta
        )
    
    def test_init_default_parameters(self):
        """Test initialization with default alpha and beta values."""
        fusion = RakingFusion(self.bm25_ranking, self.emb_rankings)
        assert fusion.alpha == 0.4
        assert fusion.beta == 0.6
        assert fusion.bm_25_ranking == self.bm25_ranking
        assert fusion.emb_rankings == self.emb_rankings
    
    def test_init_custom_parameters(self):
        """Test initialization with custom alpha and beta values."""
        custom_alpha = 0.3
        custom_beta = 0.7
        fusion = RakingFusion(
            self.bm25_ranking, 
            self.emb_rankings, 
            alpha=custom_alpha, 
            beta=custom_beta
        )
        assert fusion.alpha == custom_alpha
        assert fusion.beta == custom_beta
    
    def test_weighted_sum_basic(self):
        """Test basic weighted sum functionality."""
        result = self.fusion.weighted_sum()
        assert len(result) == 4

        doc_ids = [doc_id for doc_id, _ in result]
        expected_doc_ids = ["doc1", "doc2", "doc3", "doc4"]
        assert set(doc_ids) == set(expected_doc_ids)
    
    def test_weighted_sum_score_calculation(self):
        """Test that weighted sum scores are calculated correctly."""
        result = self.fusion.weighted_sum()
        result_dict = dict(result)
        
        # Manual calculation for verification
        # doc1: 0.4 * 0.8 + 0.6 * 0.9 = 0.32 + 0.54 = 0.86
        expected_doc1_score = self.alpha * 0.8 + self.beta * 0.9
        assert abs(result_dict["doc1"] - expected_doc1_score) < 1e-10
        
        # doc2: 0.4 * 0.6 + 0.6 * 0.3 = 0.24 + 0.18 = 0.42
        expected_doc2_score = self.alpha * 0.6 + self.beta * 0.3
        assert abs(result_dict["doc2"] - expected_doc2_score) < 1e-10
        
        # doc3: 0.4 * 0.4 + 0.6 * 0.7 = 0.16 + 0.42 = 0.58
        expected_doc3_score = self.alpha * 0.4 + self.beta * 0.7
        assert abs(result_dict["doc3"] - expected_doc3_score) < 1e-10
        
        # doc4: 0.4 * 0.2 + 0.6 * 0.5 = 0.08 + 0.30 = 0.38
        expected_doc4_score = self.alpha * 0.2 + self.beta * 0.5
        assert abs(result_dict["doc4"] - expected_doc4_score) < 1e-10
    
    def test_weighted_sum_different_alphas_betas(self):
        """Test weighted sum with different alpha and beta values."""
        # Test with equal weights
        fusion_equal = RakingFusion(
            self.bm25_ranking, 
            self.emb_rankings, 
            alpha=0.5, 
            beta=0.5
        )
        result_equal = fusion_equal.weighted_sum()
        result_dict = dict(result_equal)
        
        # doc1: 0.5 * 0.8 + 0.5 * 0.9 = 0.85
        expected_score = 0.5 * 0.8 + 0.5 * 0.9
        assert abs(result_dict["doc1"] - expected_score) < 1e-10
        
        # Test with BM25 dominance
        fusion_bm25_heavy = RakingFusion(
            self.bm25_ranking, 
            self.emb_rankings, 
            alpha=0.8, 
            beta=0.2
        )
        result_bm25 = fusion_bm25_heavy.weighted_sum()
        result_dict_bm25 = dict(result_bm25)
        
        # doc1: 0.8 * 0.8 + 0.2 * 0.9 = 0.82
        expected_score_bm25 = 0.8 * 0.8 + 0.2 * 0.9
        assert abs(result_dict_bm25["doc1"] - expected_score_bm25) < 1e-10
    
    def test_weighted_sum_partial_overlap(self):
        """Test weighted sum when not all documents overlap between rankings."""
        # BM25 ranking with doc5 that doesn't exist in embedding ranking
        bm25_partial = [
            ("doc1", 0.8),
            ("doc2", 0.6),
            ("doc5", 0.3)  # This doc doesn't exist in embedding ranking
        ]
        
        emb_partial = [
            ("doc1", 0.9),
            ("doc2", 0.3),
            ("doc6", 0.7)  # This doc doesn't exist in BM25 ranking
        ]
        
        fusion_partial = RakingFusion(bm25_partial, emb_partial)
        result = fusion_partial.weighted_sum()
        
        # Should only return documents that exist in both rankings
        assert len(result) == 2
        doc_ids = [doc_id for doc_id, _ in result]
        assert set(doc_ids) == {"doc1", "doc2"}
    
    def test_weighted_sum_empty_rankings(self):
        """Test weighted sum with empty rankings."""
        fusion_empty = RakingFusion([], [])
        result = fusion_empty.weighted_sum()
        assert result == []
    
    def test_weighted_sum_no_overlap(self):
        """Test weighted sum when there's no overlap between rankings."""
        bm25_no_overlap = [("doc1", 0.8), ("doc2", 0.6)]
        emb_no_overlap = [("doc3", 0.9), ("doc4", 0.7)]
        
        fusion_no_overlap = RakingFusion(bm25_no_overlap, emb_no_overlap)
        result = fusion_no_overlap.weighted_sum()
        assert result == []
    
    def test_weighted_sum_zero_scores(self):
        """Test weighted sum with zero scores."""
        bm25_zero = [("doc1", 0.0), ("doc2", 0.5)]
        emb_zero = [("doc1", 0.8), ("doc2", 0.0)]
        
        fusion_zero = RakingFusion(bm25_zero, emb_zero, alpha=0.4, beta=0.6)
        result = fusion_zero.weighted_sum()
        result_dict = dict(result)
        
        # doc1: 0.4 * 0.0 + 0.6 * 0.8 = 0.48
        assert abs(result_dict["doc1"] - 0.48) < 1e-10
        # doc2: 0.4 * 0.5 + 0.6 * 0.0 = 0.20
        assert abs(result_dict["doc2"] - 0.20) < 1e-10
    
    def test_weighted_sum_alpha_beta_sum_not_one(self):
        """Test that the method works even when alpha + beta != 1."""
        fusion_custom = RakingFusion(
            self.bm25_ranking, 
            self.emb_rankings, 
            alpha=0.3, 
            beta=0.8  # Sum = 1.1
        )
        result = fusion_custom.weighted_sum()
        result_dict = dict(result)
        
        # doc1: 0.3 * 0.8 + 0.8 * 0.9 = 0.24 + 0.72 = 0.96
        expected_score = 0.3 * 0.8 + 0.8 * 0.9
        assert abs(result_dict["doc1"] - expected_score) < 1e-10
    
    def test_reciprocal_rank_fusion_not_implemented(self):
        """Test that reciprocal_rank_fusion method exists but is not implemented."""
        result = self.fusion.reciprocal_rank_fusion()
        assert result is None  # Since it's not implemented yet
    
    def test_weighted_sum_preserves_order_stability(self):
        """Test that the order of results is stable and predictable."""
        # Run multiple times to ensure consistency
        result1 = self.fusion.weighted_sum()
        result2 = self.fusion.weighted_sum()
        assert result1 == result2
    
    def test_weighted_sum_type_validation(self):
        """Test that the method returns the correct types."""
        result = self.fusion.weighted_sum()
        
        # Check return type
        assert isinstance(result, list)
        
        # Check each item is a tuple of (str, float)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # document ID
            assert isinstance(item[1], (int, float))  # score


# Additional integration test
class TestRakingFusionIntegration:
    """Integration tests for RakingFusion with realistic data."""
    
    def test_realistic_search_scenario(self):
        """Test with realistic search ranking data."""
        bm25_results = [
            ("ml_intro_2023", 4.2),
            ("deep_learning_basics", 3.8),
            ("ai_overview", 2.1),
            ("neural_networks", 1.9),
            ("data_science_guide", 1.2)
        ]
        
        embedding_results = [
            ("ml_intro_2023", 0.92),
            ("neural_networks", 0.88),
            ("deep_learning_basics", 0.76),
            ("ai_overview", 0.65),
            ("data_science_guide", 0.43)
        ]
        
        fusion = RakingFusion(
            bm25_results, 
            embedding_results, 
            alpha=0.3,
            beta=0.7
        )
        
        result = fusion.weighted_sum()
        
        assert len(result) == 5
        
        result_dict = dict(result)
        for doc_id, score in result_dict.items():
            assert 0 <= score <= 10
            assert isinstance(score, (int, float))
