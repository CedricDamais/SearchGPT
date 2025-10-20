
class RakingFusion:

    def __init__(self, bm_25_ranking : list[tuple[str, float]], emb_rankings : list[tuple[str, float]], alpha : float = 0.4, beta : float = 0.6) -> None :
        self.bm_25_ranking = bm_25_ranking
        self.emb_rankings = emb_rankings
        self.alpha = alpha
        self.beta = beta
    
    def weighted_sum(self) -> list[tuple[str, float]] :
        """
        Computes a new score based on the given alpha and beta hyperparameters

        for a given document and its score:
        new_scores = alpha * bm_25_score + beta * emb_ranking_score

        If a document appears in only one ranking, use 0 for the missing score.

        Return : Final ranking list
        """
        bm25_scores = {doc_id: score for doc_id, score in self.bm_25_ranking}
        emb_scores = {doc_id: score for doc_id, score in self.emb_rankings}
        
        all_doc_ids = set(bm25_scores.keys()) | set(emb_scores.keys())
        
        final_ranking : list[tuple[str,float]] = []
        
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            emb_score = emb_scores.get(doc_id, 0.0)
            
            new_score : float = self.alpha * bm25_score + self.beta * emb_score
            final_ranking.append((doc_id, new_score))
        
        return final_ranking
    
    def reciprocal_rank_fusion(self) -> list[tuple[str, float]] :
        pass

