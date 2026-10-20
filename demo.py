#!/usr/bin/env python3
"""
Recommendation System Walkthrough & Interactive Demo
===================================================
Demonstrates:
1. Similarity Logic (Cosine, Jaccard, Euclidean, TF-IDF)
2. Real-World AI Recommendation Pipeline (Retrieval -> Ranking -> Re-ranking)
"""

import json
from pathlib import Path
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


DATA_DIR = Path(__file__).parent / "data"


def load_dataset():
    with open(DATA_DIR / "products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
    with open(DATA_DIR / "interactions.json", "r", encoding="utf-8") as f:
        interactions = json.load(f)
    return products, interactions


def demo_similarity_math():
    print("=" * 80)
    print("  PART 1: SIMILARITY LOGIC DEEP DIVE")
    print("=" * 80)

    # 1. Cosine vs Euclidean
    vec_a = [1.0, 2.0, 0.0, 4.0]
    vec_b = [2.0, 4.0, 0.0, 8.0]  # Exact same direction, double magnitude
    vec_c = [0.0, 0.0, 3.0, 0.0]  # Completely orthogonal (unrelated)

    cos_ab = cosine_similarity(vec_a, vec_b)
    cos_ac = cosine_similarity(vec_a, vec_c)
    euc_dist_ab = euclidean_distance(vec_a, vec_b)

    print("\n1. Geometric Vector Comparison:")
    print(f"   Vector A: {vec_a}")
    print(f"   Vector B: {vec_b} (identical direction as A, but 2x length)")
    print(f"   Vector C: {vec_c} (orthogonal to A)")
    print(f"   -> Cosine Similarity(A, B) = {cos_ab:.4f} (1.0 = identical profile!)")
    print(f"   -> Euclidean Distance(A, B) = {euc_dist_ab:.4f} (affected by scale)")
    print(f"   -> Cosine Similarity(A, C) = {cos_ac:.4f} (0.0 = completely unrelated)")

    # 2. Jaccard Similarity on Product Tags
    tags_sony = {"wireless", "noise-canceling", "over-ear", "bluetooth", "travel", "audio"}
    tags_bose = {"wireless", "noise-canceling", "over-ear", "bluetooth", "travel", "audio"}
    tags_airpods = {"wireless", "noise-canceling", "in-ear", "bluetooth", "compact", "spatial-audio"}
    tags_macbook = {"laptop", "macos", "m3", "developer", "creator", "retina"}

    jaccard_sony_bose = jaccard_similarity(tags_sony, tags_bose)
    jaccard_sony_airpods = jaccard_similarity(tags_sony, tags_airpods)
    jaccard_sony_macbook = jaccard_similarity(tags_sony, tags_macbook)

    print("\n2. Jaccard Tag Similarity (Intersection over Union):")
    print(f"   Sony Over-Ear Tags: {tags_sony}")
    print(f"   Bose Over-Ear Tags: {tags_bose}")
    print(f"   AirPods In-Ear Tags: {tags_airpods}")
    print(f"   MacBook Tags:       {tags_macbook}")
    print(f"   -> Jaccard(Sony, Bose):    {jaccard_sony_bose:.4f} (100% tag match)")
    print(f"   -> Jaccard(Sony, AirPods): {jaccard_sony_airpods:.4f} (partial match)")
    print(f"   -> Jaccard(Sony, MacBook): {jaccard_sony_macbook:.4f} (0% match)")

    # 3. TF-IDF Representation
    sample_texts = [
        "Noise cancelling wireless over ear headphones for travel.",
        "Wireless in ear bluetooth earbuds with high fidelity audio.",
        "Pro laptop with retina screen and high performance chip.",
    ]
    vectorizer = TFIDFVectorizer().fit(sample_texts)
    vecs = vectorizer.transform(sample_texts)
    text_sim = cosine_similarity(vecs[0], vecs[1])
    laptop_sim = cosine_similarity(vecs[0], vecs[2])

    print("\n3. TF-IDF Textual Cosine Similarity:")
    print(f"   Doc 1: '{sample_texts[0]}'")
    print(f"   Doc 2: '{sample_texts[1]}'")
    print(f"   Doc 3: '{sample_texts[2]}'")
    print(f"   -> Cosine Sim(Doc 1, Doc 2): {text_sim:.4f} (both audio/wireless)")
    print(f"   -> Cosine Sim(Doc 1, Doc 3): {laptop_sim:.4f} (minimal overlap)")


def demo_real_world_pipeline():
    print("\n" + "=" * 80)
    print("  PART 2: REAL-WORLD AI RECOMMENDATION PIPELINE IN ACTION")
    print("=" * 80)

    products, interactions = load_dataset()

    # Initialize engines
    content_rec = ContentBasedRecommender().fit(products)
    collab_rec = CollaborativeRecommender().fit(interactions)
    pipeline = RecommendationPipeline(content_rec, collab_rec)

    # Scenario: User is viewing product prod_001 (Sony WH-1000XM5 Headphones)
    target_id = "prod_001"
    target = content_rec.id_to_product[target_id]

    print(f"\nUser is viewing: [{target.id}] {target.title}")
    print(f"Category: {target.category} | Price: ${target.price:.2f} | Rating: {target.rating}★")

    response = pipeline.recommend(
        target_product_id=target_id,
        top_k=4,
        filter_out_of_stock=True,
        max_per_category=2,  # Ensures diversity!
    )

    summary = response["stages_summary"]
    print("\n--- Funnel Progression ---")
    print(f"1. Candidate Retrieval: Filtered entire catalog -> {summary['stage_1_candidates_count']} candidates")
    print(f"2. Multi-Factor Scoring: Ranked {summary['stage_2_ranked_count']} candidates")
    print(f"3. Re-Ranking & Diversity: Produced Top {summary['stage_3_final_slate_count']} final recommendation slate")

    print("\n--- Final Recommended Slate ---")
    for rec in response["recommendations"]:
        print(f"\n[#{rec['rank']}] {rec['title']}")
        print(f"     Category: {rec['category']} | Price: ${rec['price']} | Rating: {rec['rating']}★")
        print(f"     Overall Rank Score: {rec['final_rank_score']:.4f}")
        print(f"     Channel: {rec['retrieval_source']}")
        print(f"     Content Similarity: {rec['content_similarity']:.4f}")
        print(f"     Collaborative Score: {rec['collaborative_score']:.4f}")
        if "collab_reason" in rec["stage_notes"]:
            print(f"     Social Proof: {rec['stage_notes']['collab_reason']}")


def demo_collaborative_cross_sell():
    print("\n" + "=" * 80)
    print("  PART 3: COLLABORATIVE FILTERING & CROSS-SELLING")
    print("=" * 80)

    products, interactions = load_dataset()
    collab_rec = CollaborativeRecommender().fit(interactions)

    macbook_id = "prod_009"
    collab_items = collab_rec.recommend_for_item(macbook_id, top_k=3)

    print(f"\nItem-Item Collaborative Discovery for [Apple MacBook Pro 14 M3]:")
    print("Notice how collaborative filtering surfaces accessories (Mice & Keyboards) that")
    print("content-based filtering alone would miss due to different categories!\n")

    id_to_title = {p["id"]: p["title"] for p in products}
    for match in collab_items:
        title = id_to_title.get(match.product_id, match.product_id)
        print(f"  -> {title}")
        print(f"     Co-interaction score: {match.score:.4f} | {match.reason}")


if __name__ == "__main__":
    demo_similarity_math()
    demo_real_world_pipeline()
    demo_collaborative_cross_sell()

