
class RakingFusion:

    def __init__(self, bm_25_ranking : list[tuple[str, float]], emb_rankings : list[tuple[str, float]], alpha : float = 0.5, beta : float = 0.5) -> None :
        self.bm_25_ranking = bm_25_ranking
        self.emb_rankings = emb_rankings
        self.alpha = alpha
        self.beta = beta
    
    def weighted_sum(self) -> list[tuple[str, float]] :
        """
        Computes a new score based on the given alpha and beta hyperparameters

        for a given document and its score:
        new_scores = alpha * bm_25_score + beta * emb_ranking_score

        Return : Final ranking list
        """
        final_ranking : list[tuple[str,float]] = []

        for document, score in self.bm_25_ranking:

            for emb_document, emb_score in self.emb_rankings:

                if emb_document == document:
                    new_score : float = self.alpha * score + self.beta * emb_score
                    final_ranking.append((document, new_score))
        return final_ranking
    
    def reciprocal_rank_fusion(self) -> list[tuple[str, float]] :
        pass
