"""
Collaborative Filtering Engine
==============================
Discovers collective behavioral patterns:
- "Users who bought this also bought..." (Item-Item Collaborative Filtering)
- "Recommended for you based on similar users" (User-Item Collaborative Filtering)
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple
from .similarity import cosine_similarity


@dataclass
class CollaborativeMatch:
    product_id: str
    score: float
    shared_users_count: int
    reason: str


class CollaborativeRecommender:
    """
    Implements memory-based Item-to-Item and User-to-Item collaborative filtering.
    """

    def __init__(self):
        # user_id -> {product_id: weight}
        self.user_item_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        # product_id -> {user_id: weight}
        self.item_user_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.all_users: List[str] = []
        self.all_products: List[str] = []

    def fit(self, interactions: List[Dict[str, Any]]) -> "CollaborativeRecommender":
        """Builds user-item and item-user interaction matrices."""
        self.user_item_matrix.clear()
        self.item_user_matrix.clear()

        user_set: Set[str] = set()
        prod_set: Set[str] = set()

        for inter in interactions:
            u = str(inter["user_id"])
            p = str(inter["product_id"])
            w = float(inter.get("weight", 1.0))

            # Store the highest interaction weight if duplicates exist
            if w > self.user_item_matrix[u].get(p, 0.0):
                self.user_item_matrix[u][p] = w
            if w > self.item_user_matrix[p].get(u, 0.0):
                self.item_user_matrix[p][u] = w

            user_set.add(u)
            prod_set.add(p)

        self.all_users = sorted(user_set)
        self.all_products = sorted(prod_set)
        return self

    def _get_item_vector(self, product_id: str) -> List[float]:
        """Constructs vector of user interaction weights for a product."""
        user_ratings = self.item_user_matrix.get(product_id, {})
        return [user_ratings.get(u, 0.0) for u in self.all_users]

    def _get_user_vector(self, user_id: str) -> List[float]:
        """Constructs vector of product interaction weights for a user."""
        prod_ratings = self.user_item_matrix.get(user_id, {})
        return [prod_ratings.get(p, 0.0) for p in self.all_products]

    def item_similarity(self, item_a: str, item_b: str) -> Tuple[float, int]:
        """
        Calculates cosine similarity between two items across all users.
        Also returns the count of users who interacted with both.
        """
        users_a = set(self.item_user_matrix.get(item_a, {}).keys())
        users_b = set(self.item_user_matrix.get(item_b, {}).keys())
        shared = users_a.intersection(users_b)

        if not shared:
            return 0.0, 0

        vec_a = self._get_item_vector(item_a)
        vec_b = self._get_item_vector(item_b)
        sim = cosine_similarity(vec_a, vec_b)
        return sim, len(shared)

    def recommend_for_item(self, product_id: str, top_k: int = 5) -> List[CollaborativeMatch]:
        """
        'Customers Who Bought This Also Bought':
        Finds products most frequently co-interacted with the given item.
        """
        results: List[CollaborativeMatch] = []

        if product_id not in self.item_user_matrix:
            return []

        for candidate_id in self.all_products:
            if candidate_id == product_id:
                continue

            sim, shared_count = self.item_similarity(product_id, candidate_id)
            if sim > 0.0:
                results.append(
                    CollaborativeMatch(
                        product_id=candidate_id,
                        score=round(sim, 4),
                        shared_users_count=shared_count,
                        reason=f"{shared_count} customers interacted with both items",
                    )
                )

        results.sort(key=lambda x: (x.score, x.shared_users_count), reverse=True)
        return results[:top_k]

    def recommend_for_user(self, user_id: str, top_k: int = 5) -> List[CollaborativeMatch]:
        """
        User-based Collaborative Filtering:
        1. Find most similar users based on interaction vector cosine similarity.
        2. Aggregate candidate items that similar users bought, weighted by user similarity.
        """
        if user_id not in self.user_item_matrix:
            return []

        user_vec = self._get_user_vector(user_id)
        already_interacted = set(self.user_item_matrix[user_id].keys())

        # 1. Compute similarity with every other user
        user_similarities: List[Tuple[str, float]] = []
        for other_u in self.all_users:
            if other_u == user_id:
                continue
            other_vec = self._get_user_vector(other_u)
            sim = cosine_similarity(user_vec, other_vec)
            if sim > 0:
                user_similarities.append((other_u, sim))

        # 2. Score candidate items
        candidate_scores: Dict[str, float] = defaultdict(float)
        candidate_counts: Dict[str, int] = defaultdict(int)

        for other_u, u_sim in user_similarities:
            for prod_id, weight in self.user_item_matrix[other_u].items():
                if prod_id in already_interacted:
                    continue
                candidate_scores[prod_id] += u_sim * weight
                candidate_counts[prod_id] += 1

        matches: List[CollaborativeMatch] = []
        for prod_id, score in candidate_scores.items():
            matches.append(
                CollaborativeMatch(
                    product_id=prod_id,
                    score=round(score, 4),
                    shared_users_count=candidate_counts[prod_id],
                    reason=f"Recommended by {candidate_counts[prod_id]} similar shoppers",
                )
            )

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:top_k]

