"""
Content-Based Recommendation Engine
===================================
Suggests products based on item attributes:
- Text content (Title + Description) via TF-IDF & Cosine Similarity
- Categorical attributes & tags via Jaccard Similarity
- Price neighborhood proximity
- Category alignment
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .similarity import TFIDFVectorizer, cosine_similarity, jaccard_similarity


@dataclass
class Product:
    id: str
    title: str
    category: str
    brand: str
    price: float
    rating: float
    reviews_count: int
    in_stock: bool
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def get_full_text(self) -> str:
        """Combines textual properties into a rich document for text vectorization."""
        tags_str = " ".join(self.tags)
        return f"{self.title} {self.brand} {self.category} {tags_str} {self.description}"


@dataclass
class RecommendationExplanation:
    product: Product
    total_score: float
    text_similarity: float
    tag_similarity: float
    category_match: float
    price_closeness: float
    breakdown: Dict[str, float] = field(default_factory=dict)


class ContentBasedRecommender:
    def __init__(
        self,
        weight_text: float = 0.45,
        weight_tags: float = 0.35,
        weight_category: float = 0.10,
        weight_price: float = 0.10,
    ):
        self.w_text = weight_text
        self.w_tags = weight_tags
        self.w_cat = weight_category
        self.w_price = weight_price

        self.products: List[Product] = []
        self.id_to_product: Dict[str, Product] = {}
        self.id_to_index: Dict[str, int] = {}
        self.vectorizer = TFIDFVectorizer()
        self.tfidf_vectors: List[List[float]] = []

    def fit(self, raw_products: List[Dict[str, Any]]) -> "ContentBasedRecommender":
        """Loads and indexes the catalog."""
        self.products = [
            Product(
                id=p["id"],
                title=p["title"],
                category=p["category"],
                brand=p["brand"],
                price=float(p["price"]),
                rating=float(p.get("rating", 0.0)),
                reviews_count=int(p.get("reviews_count", 0)),
                in_stock=bool(p.get("in_stock", True)),
                tags=list(p.get("tags", [])),
                description=p.get("description", ""),
            )
            for p in raw_products
        ]
        self.id_to_product = {p.id: p for p in self.products}
        self.id_to_index = {p.id: i for i, p in enumerate(self.products)}

        # Build TF-IDF representations for all products
        documents = [p.get_full_text() for p in self.products]
        self.tfidf_vectors = self.vectorizer.fit_transform(documents)
        return self

    def compute_similarity(self, target: Product, candidate: Product) -> RecommendationExplanation:
        """Computes comprehensive multi-attribute similarity between target and candidate."""
        idx_target = self.id_to_index[target.id]
        idx_candidate = self.id_to_index[candidate.id]

        # 1. Text Similarity (Cosine Similarity over TF-IDF vectors)
        vec_target = self.tfidf_vectors[idx_target]
        vec_candidate = self.tfidf_vectors[idx_candidate]
        sim_text = cosine_similarity(vec_target, vec_candidate)

        # 2. Tag Similarity (Jaccard Similarity)
        sim_tags = jaccard_similarity(target.tags, candidate.tags)

        # 3. Category Match (Binary match 1.0 or 0.0)
        sim_cat = 1.0 if target.category.lower() == candidate.category.lower() else 0.0

        # 4. Price Closeness: 1 - |p1 - p2| / max(p1, p2)
        max_price = max(target.price, candidate.price, 1e-6)
        price_diff = abs(target.price - candidate.price)
        sim_price = max(0.0, 1.0 - (price_diff / max_price))

        total_score = (
            (self.w_text * sim_text)
            + (self.w_tags * sim_tags)
            + (self.w_cat * sim_cat)
            + (self.w_price * sim_price)
        )

        return RecommendationExplanation(
            product=candidate,
            total_score=round(total_score, 4),
            text_similarity=round(sim_text, 4),
            tag_similarity=round(sim_tags, 4),
            category_match=round(sim_cat, 4),
            price_closeness=round(sim_price, 4),
            breakdown={
                "w_text_contrib": round(self.w_text * sim_text, 4),
                "w_tags_contrib": round(self.w_tags * sim_tags, 4),
                "w_cat_contrib": round(self.w_cat * sim_cat, 4),
                "w_price_contrib": round(self.w_price * sim_price, 4),
            },
        )

    def recommend_similar_to_product(
        self, product_id: str, top_k: int = 5, only_in_stock: bool = True
    ) -> List[RecommendationExplanation]:
        """
        'Suggest Products Like This':
        Finds the top-K most similar items to the target item.
        """
        if product_id not in self.id_to_product:
            raise KeyError(f"Product '{product_id}' not found in catalog.")

        target = self.id_to_product[product_id]
        scores: List[RecommendationExplanation] = []

        for candidate in self.products:
            if candidate.id == target.id:
                continue
            if only_in_stock and not candidate.in_stock:
                continue

            explanation = self.compute_similarity(target, candidate)
            scores.append(explanation)

        # Sort descending by total similarity score
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores[:top_k]

