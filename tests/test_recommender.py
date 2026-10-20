"""
Unit Tests for Recommendation System
====================================
Verifies:
1. Similarity metrics (Cosine, Jaccard, Euclidean)
2. TF-IDF vectorization and normalization
3. Content-based recommendation behavior
4. Collaborative filtering co-occurrence scoring
5. Multi-stage pipeline (Retrieval -> Ranking -> Re-ranking with out-of-stock and diversity constraints)
"""

import json
from pathlib import Path
import unittest

from src.similarity import (
    cosine_similarity,
    jaccard_similarity,
    euclidean_distance,
    euclidean_similarity,
    TFIDFVectorizer,
)
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.pipeline import RecommendationPipeline


class TestSimilarityMetrics(unittest.TestCase):
    def test_cosine_identical_direction(self):
        v1 = [1.0, 2.0, 3.0]
        v2 = [2.0, 4.0, 6.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, places=5)

    def test_cosine_orthogonal(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 5.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0, places=5)

    def test_cosine_zero_vector(self):
        v1 = [0.0, 0.0]
        v2 = [1.0, 2.0]
        self.assertEqual(cosine_similarity(v1, v2), 0.0)

    def test_jaccard_identical_and_disjoint(self):
        s1 = {"a", "b", "c"}
        s2 = {"a", "b", "c"}
        self.assertEqual(jaccard_similarity(s1, s2), 1.0)

        s3 = {"x", "y"}
        self.assertEqual(jaccard_similarity(s1, s3), 0.0)

        s4 = {"b", "c", "d"}
        # intersection = 2 (b, c), union = 4 (a, b, c, d) -> 0.5
        self.assertEqual(jaccard_similarity(s1, s4), 0.5)

    def test_euclidean_metrics(self):
        p1 = [0.0, 0.0]
        p2 = [3.0, 4.0]
        self.assertAlmostEqual(euclidean_distance(p1, p2), 5.0, places=5)
        self.assertAlmostEqual(euclidean_similarity(p1, p2), 1.0 / 6.0, places=5)


class TestTFIDFVectorizer(unittest.TestCase):
    def test_tfidf_vectors(self):
        docs = [
            "wireless noise canceling headphones",
            "wireless ergonomic keyboard and mouse",
            "travel noise canceling headphones",
        ]
        vectorizer = TFIDFVectorizer().fit(docs)
        vecs = vectorizer.transform(docs)
        self.assertEqual(len(vecs), 3)

        # Docs 0 and 2 share "noise", "canceling", "headphones"
        sim_0_2 = cosine_similarity(vecs[0], vecs[2])
        sim_0_1 = cosine_similarity(vecs[0], vecs[1])
        self.assertGreater(sim_0_2, sim_0_1)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        data_dir = Path(__file__).parent.parent / "data"
        with open(data_dir / "products.json", "r", encoding="utf-8") as f:
            self.products = json.load(f)
        with open(data_dir / "interactions.json", "r", encoding="utf-8") as f:
            self.interactions = json.load(f)

        self.content_rec = ContentBasedRecommender().fit(self.products)
        self.collab_rec = CollaborativeRecommender().fit(self.interactions)
        self.pipeline = RecommendationPipeline(self.content_rec, self.collab_rec)

    def test_out_of_stock_filtering(self):
        # prod_011 is Lenovo ThinkPad X1 Carbon Gen 11 (in_stock: false)
        res = self.pipeline.recommend("prod_009", top_k=10, filter_out_of_stock=True)
        recommended_ids = [r["id"] for r in res["recommendations"]]
        self.assertNotIn("prod_011", recommended_ids)

    def test_category_diversity_constraint(self):
        # Setting max_per_category to 1 should guarantee unique categories in top slots
        res = self.pipeline.recommend(
            "prod_001", top_k=3, filter_out_of_stock=True, max_per_category=1
        )
        cats = [r["category"] for r in res["recommendations"]]
        self.assertEqual(len(cats), len(set(cats)))

    def test_collaborative_cross_sell(self):
        matches = self.collab_rec.recommend_for_item("prod_009", top_k=3)
        match_ids = [m.product_id for m in matches]
        # MacBook buyers interacted with accessories
        self.assertTrue(any(pid in match_ids for pid in ["prod_013", "prod_014", "prod_015"]))


if __name__ == "__main__":
    unittest.main()

