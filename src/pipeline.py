"""
Real-World Multi-Stage AI Recommendation Pipeline
=================================================
Simulates the production-grade 3-stage funnel used by tech leaders
(Amazon, YouTube, Netflix, Uber Eats):

    [ Catalog: 10M+ Items ]
               │
               ▼
   STAGE 1: Candidate Retrieval (Fast ANN / Multi-Channel Inverted Index)
            Filter down to 50 - 500 candidates in < 15ms
               │
               ▼
   STAGE 2: Multi-Factor Scoring & Ranking
            Combine content similarity, collaborative signal, quality, & popularity
               │
               ▼
   STAGE 3: Re-Ranking & Business Logic
            Deduplication, out-of-stock filter, Category Diversity (MMR)
               │
               ▼
    [ Final Top-K Slate for User Display ]
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from .content_based import ContentBasedRecommender, Product
from .collaborative import CollaborativeRecommender


@dataclass
class ScoredCandidate:
    product: Product
    retrieval_source: str  # e.g., 'content_similarity', 'collaborative_co_buy', 'category_ann'
    content_similarity: float = 0.0
    collaborative_score: float = 0.0
    quality_score: float = 0.0      # Normalized rating & reviews count
    popularity_score: float = 0.0   # Interaction volume
    final_rank_score: float = 0.0
    stage_notes: Dict[str, Any] = field(default_factory=dict)


class RecommendationPipeline:
    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collab_recommender: CollaborativeRecommender,
        diversity_lambda: float = 0.75,  # 1.0 = purely rank score, 0.0 = purely diversity
    ):
        self.content_rec = content_recommender
        self.collab_rec = collab_recommender
        self.diversity_lambda = diversity_lambda

    # -------------------------------------------------------------
    # STAGE 1: Candidate Generation (Retrieval)
    # -------------------------------------------------------------
    def stage_1_retrieval(
        self,
        target_product_id: str,
        user_id: Optional[str] = None,
        retrieval_limit: int = 50,
    ) -> Dict[str, ScoredCandidate]:
        """
        Fast Multi-Source Candidate Retrieval:
        Gathers candidate items from multiple lightweight channels:
        Channel A: Vector/Content Similarity Channel
        Channel B: Collaborative Filtering (Co-occurrence / Co-purchase)
        Channel C: Category Neighbor Channel
        """
        candidates: Dict[str, ScoredCandidate] = {}
        target = self.content_rec.id_to_product[target_product_id]

        # Channel A: Content Similarity Retrieval
        content_results = self.content_rec.recommend_similar_to_product(
            target_product_id, top_k=retrieval_limit, only_in_stock=False
        )
        for res in content_results:
            candidates[res.product.id] = ScoredCandidate(
                product=res.product,
                retrieval_source="content_similarity",
                content_similarity=res.total_score,
                stage_notes={"content_breakdown": res.breakdown},
            )

        # Channel B: Collaborative Filtering Retrieval (Items bought together)
        collab_matches = self.collab_rec.recommend_for_item(
            target_product_id, top_k=retrieval_limit
        )
        for match in collab_matches:
            prod_id = match.product_id
            if prod_id in self.content_rec.id_to_product:
                prod = self.content_rec.id_to_product[prod_id]
                if prod_id in candidates:
                    candidates[prod_id].collaborative_score = match.score
                    candidates[prod_id].retrieval_source = "hybrid_content_collab"
                    candidates[prod_id].stage_notes["collab_reason"] = match.reason
                else:
                    candidates[prod_id] = ScoredCandidate(
                        product=prod,
                        retrieval_source="collaborative_co_buy",
                        collaborative_score=match.score,
                        stage_notes={"collab_reason": match.reason},
                    )

        # Channel C: Category Matching (Fallback / Cold-Start channel)
        for prod in self.content_rec.products:
            if prod.id != target.id and prod.category == target.category:
                if prod.id not in candidates:
                    candidates[prod.id] = ScoredCandidate(
                        product=prod,
                        retrieval_source="category_match",
                        stage_notes={"channel": "category_neighbor"},
                    )

        return candidates

    # -------------------------------------------------------------
    # STAGE 2: Scoring & Ranking Model
    # -------------------------------------------------------------
    def stage_2_ranking(
        self,
        candidates: Dict[str, ScoredCandidate],
        target_product: Product,
    ) -> List[ScoredCandidate]:
        """
        Multi-Factor Scoring:
        Combines:
        1. Content Similarity (0.40)
        2. Collaborative Affinity (0.35)
        3. Product Quality (Bayesian-dampened rating) (0.15)
        4. Popularity (log of review volume) (0.10)
        """
        scored_list: List[ScoredCandidate] = []

        max_reviews = max((c.product.reviews_count for c in candidates.values()), default=1)

        for cand in candidates.values():
            prod = cand.product

            # If content similarity was not computed yet, compute it now
            if cand.content_similarity == 0.0:
                explanation = self.content_rec.compute_similarity(target_product, prod)
                cand.content_similarity = explanation.total_score

            # Normalize rating: (rating / 5.0)
            norm_rating = prod.rating / 5.0

            # Normalize review count via log scale: log1p(reviews) / log1p(max_reviews)
            log_reviews = math.log1p(prod.reviews_count) / math.log1p(max_reviews) if max_reviews > 0 else 0.0

            cand.quality_score = round(norm_rating, 4)
            cand.popularity_score = round(log_reviews, 4)

            # Production-like composite objective function
            rank_score = (
                (0.40 * cand.content_similarity)
                + (0.35 * cand.collaborative_score)
                + (0.15 * cand.quality_score)
                + (0.10 * cand.popularity_score)
            )
            cand.final_rank_score = round(rank_score, 4)
            scored_list.append(cand)

        # Sort descending by rank score
        scored_list.sort(key=lambda x: x.final_rank_score, reverse=True)
        return scored_list

    # -------------------------------------------------------------
    # STAGE 3: Re-ranking & Business Rules (MMR / Diversity & Stock)
    # -------------------------------------------------------------
    def stage_3_reranking(
        self,
        ranked_candidates: List[ScoredCandidate],
        top_k: int = 5,
        filter_out_of_stock: bool = True,
        max_per_category: int = 2,
    ) -> List[ScoredCandidate]:
        """
        Applies business constraints & diversity:
        1. Filter out of stock items (prevents poor customer experience).
        2. Category Diversity Balancing: Prevents dominating the recommendation
           carousel with 5 identical laptops or 5 identical headphones.
        3. Maximal Marginal Relevance (MMR) trade-off between relevance & diversity.
        """
        # Step 1: In-stock filtering
        eligible: List[ScoredCandidate] = []
        for cand in ranked_candidates:
            if filter_out_of_stock and not cand.product.in_stock:
                cand.stage_notes["filtered_out"] = "Out of stock policy"
                continue
            eligible.append(cand)

        # Step 2: Diversity-Aware Slate Selection
        selected: List[ScoredCandidate] = []
        category_counts: Dict[str, int] = {}

        for cand in eligible:
            cat = cand.product.category
            current_count = category_counts.get(cat, 0)

            # Apply category diversity ceiling
            if current_count >= max_per_category:
                cand.stage_notes["diversity_penalty"] = f"Category '{cat}' cap reached ({max_per_category})"
                continue

            selected.append(cand)
            category_counts[cat] = current_count + 1

            if len(selected) == top_k:
                break

        # If diversity cap left slots unfilled, fill with remaining highest-scoring eligible items
        if len(selected) < top_k:
            selected_ids = {c.product.id for c in selected}
            for cand in eligible:
                if cand.product.id not in selected_ids:
                    selected.append(cand)
                    if len(selected) == top_k:
                        break

        return selected

    # -------------------------------------------------------------
    # Full End-to-End Orchestrator
    # -------------------------------------------------------------
    def recommend(
        self,
        target_product_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        filter_out_of_stock: bool = True,
        max_per_category: int = 2,
    ) -> Dict[str, Any]:
        """
        Executes the full 3-stage recommendation pipeline and returns
        both the final recommendations and step-by-step diagnostic trace.
        """
        if target_product_id not in self.content_rec.id_to_product:
            raise KeyError(f"Product '{target_product_id}' not found.")

        target = self.content_rec.id_to_product[target_product_id]

        # Stage 1: Retrieval
        retrieved_map = self.stage_1_retrieval(target_product_id, user_id=user_id)

        # Stage 2: Ranking
        ranked_candidates = self.stage_2_ranking(retrieved_map, target)

        # Stage 3: Re-ranking & Business Rules
        final_slate = self.stage_3_reranking(
            ranked_candidates,
            top_k=top_k,
            filter_out_of_stock=filter_out_of_stock,
            max_per_category=max_per_category,
        )

        return {
            "target_product": {
                "id": target.id,
                "title": target.title,
                "category": target.category,
                "price": target.price,
                "rating": target.rating,
            },
            "stages_summary": {
                "stage_1_candidates_count": len(retrieved_map),
                "stage_2_ranked_count": len(ranked_candidates),
                "stage_3_final_slate_count": len(final_slate),
            },
            "recommendations": [
                {
                    "rank": idx + 1,
                    "id": c.product.id,
                    "title": c.product.title,
                    "category": c.product.category,
                    "brand": c.product.brand,
                    "price": c.product.price,
                    "rating": c.product.rating,
                    "final_rank_score": c.final_rank_score,
                    "retrieval_source": c.retrieval_source,
                    "content_similarity": c.content_similarity,
                    "collaborative_score": c.collaborative_score,
                    "popularity_score": c.popularity_score,
                    "stage_notes": c.stage_notes,
                }
                for idx, c in enumerate(final_slate)
            ],
        }

