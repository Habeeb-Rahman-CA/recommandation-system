#!/usr/bin/env python3
"""
Recommendation System Web Application & REST API
================================================
Interactive visual dashboard to explore:
- Similarity Logic breakdown between products
- Real-world AI Multi-stage pipeline (Retrieval -> Ranking -> Re-ranking)
- Live recommendation suggestions for any product
"""

import json
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request

from src.similarity import (
    cosine_similarity,
    jaccard_similarity,
    euclidean_distance,
)
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.pipeline import RecommendationPipeline

app = Flask(__name__)
DATA_DIR = Path(__file__).parent / "data"

# Load data and initialize recommender engines
with open(DATA_DIR / "products.json", "r", encoding="utf-8") as f:
    products_data = json.load(f)

with open(DATA_DIR / "interactions.json", "r", encoding="utf-8") as f:
    interactions_data = json.load(f)

content_engine = ContentBasedRecommender().fit(products_data)
collab_engine = CollaborativeRecommender().fit(interactions_data)
pipeline = RecommendationPipeline(content_engine, collab_engine)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Recommendation Engine & Similarity Logic Lab</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --accent: #58a6ff;
      --accent-hover: #1f6feb;
      --text: #c9d1d9;
      --text-muted: #8b949e;
      --success: #3fb950;
      --warning: #d29922;
      --purple: #bc8cff;
      --badge-bg: #21262d;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .header {
      max-width: 1200px;
      margin: 0 auto 24px auto;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }
    .header h1 {
      font-size: 1.8rem;
      color: #f0f6fc;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .header p {
      color: var(--text-muted);
      margin-top: 6px;
      font-size: 0.95rem;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 350px 1fr;
      gap: 24px;
    }
    @media (max-width: 860px) {
      .container { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
    }
    .card h2 {
      font-size: 1.1rem;
      color: #f0f6fc;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .catalog-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 600px;
      overflow-y: auto;
    }
    .product-item {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .product-item:hover {
      border-color: var(--accent);
      transform: translateY(-1px);
    }
    .product-item.active {
      border-color: var(--accent);
      background: #1f2937;
    }
    .item-header {
      display: flex;
      justify-content: space-between;
      font-weight: 600;
      font-size: 0.9rem;
      color: #f0f6fc;
    }
    .item-meta {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 4px;
      display: flex;
      gap: 8px;
    }
    .badge {
      display: inline-block;
      padding: 2px 6px;
      font-size: 0.75rem;
      border-radius: 4px;
      background: var(--badge-bg);
      border: 1px solid var(--border);
      color: var(--text);
    }
    .badge.cat { color: var(--accent); }
    .badge.brand { color: var(--purple); }
    .badge.rating { color: var(--warning); }
    .badge.out-of-stock { color: #f85149; border-color: #f85149; }

    .controls {
      display: flex;
      gap: 16px;
      background: #0d1117;
      border: 1px solid var(--border);
      padding: 12px 16px;
      border-radius: 6px;
      margin-bottom: 16px;
      align-items: center;
      flex-wrap: wrap;
    }
    .control-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
    }
    .pipeline-stage {
      display: flex;
      justify-content: space-between;
      background: #0d1117;
      border: 1px solid var(--border);
      padding: 12px;
      border-radius: 6px;
      margin-bottom: 16px;
      font-size: 0.85rem;
    }
    .stage-step {
      text-align: center;
      flex: 1;
    }
    .stage-step .count {
      font-size: 1.2rem;
      font-weight: bold;
      color: var(--accent);
    }
    .rec-grid {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .rec-card {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .rec-card-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }
    .rank-num {
      background: var(--accent);
      color: #0d1117;
      font-weight: 700;
      border-radius: 4px;
      padding: 2px 8px;
      font-size: 0.85rem;
      margin-right: 8px;
    }
    .score-breakdown {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
      margin-top: 6px;
      padding-top: 8px;
      border-top: 1px dashed var(--border);
    }
    .metric-pill {
      background: var(--card-bg);
      padding: 6px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
    }
    .metric-pill .label { color: var(--text-muted); display: block; }
    .metric-pill .val { font-weight: 600; color: #f0f6fc; }
  </style>
</head>
<body>

  <div class="header">
    <h1>🛍️ AI Product Recommendation Engine</h1>
    <p>Explore First-Principles <strong>Similarity Logic</strong> &amp; the <strong>3-Stage Real-World AI Architecture</strong> (Retrieval → Ranking → Re-Ranking)</p>
  </div>

  <div class="container">
    <!-- Left Column: Product Catalog -->
    <div>
      <div class="card">
        <h2>📦 Product Catalog <span class="badge" id="productCount"></span></h2>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 12px;">
          Select any product to inspect real-time recommendations:
        </p>
        <div class="catalog-list" id="catalogList"></div>
      </div>
    </div>

    <!-- Right Column: Recommendation Pipeline Results -->
    <div>
      <div class="card">
        <h2>🎯 Active Target Product</h2>
        <div id="targetDisplay" style="color: var(--text-muted); font-size: 0.9rem;">
          Select an item from the left.
        </div>
      </div>

      <div class="card">
        <h2>⚙️ Real-World AI Funnel Pipeline</h2>
        
        <div class="controls">
          <label class="control-item">
            <input type="checkbox" id="stockFilter" checked onchange="loadRecommendations()">
            Filter Out-of-Stock
          </label>
          <label class="control-item">
            Category Diversity Cap:
            <select id="categoryCap" onchange="loadRecommendations()" style="background:#161b22; color:#fff; border:1px solid #30363d; padding:2px 6px; border-radius:4px;">
              <option value="1">Strict (Max 1 per category)</option>
              <option value="2" selected>Balanced (Max 2 per category)</option>
              <option value="99">Off (No diversity limit)</option>
            </select>
          </label>
        </div>

        <div class="pipeline-stage" id="pipelineStats">
          <div class="stage-step">
            <div class="count" id="statRetrieval">0</div>
            <div>Stage 1: Retrieval</div>
          </div>
          <div style="align-self: center; color: var(--text-muted);">➔</div>
          <div class="stage-step">
            <div class="count" id="statRanking">0</div>
            <div>Stage 2: Ranking</div>
          </div>
          <div style="align-self: center; color: var(--text-muted);">➔</div>
          <div class="stage-step">
            <div class="count" id="statReranking">0</div>
            <div>Stage 3: Re-Ranking</div>
          </div>
        </div>

        <h3 style="font-size: 1rem; color: #f0f6fc; margin-bottom: 12px;">Recommended Products (Top Slate)</h3>
        <div class="rec-grid" id="recommendationsList">
          <p style="color: var(--text-muted); font-size: 0.9rem;">No product selected.</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentProductId = "prod_001";
    let catalog = [];

    async function init() {
      const res = await fetch('/api/products');
      catalog = await res.json();
      document.getElementById('productCount').textContent = catalog.length;
      renderCatalog();
      selectProduct(currentProductId);
    }

    function renderCatalog() {
      const container = document.getElementById('catalogList');
      container.innerHTML = catalog.map(p => `
        <div class="product-item ${p.id === currentProductId ? 'active' : ''}" onclick="selectProduct('${p.id}')">
          <div class="item-header">
            <span>${p.title}</span>
            <span>$${p.price.toFixed(2)}</span>
          </div>
          <div class="item-meta">
            <span class="badge cat">${p.category}</span>
            <span class="badge brand">${p.brand}</span>
            <span class="badge rating">★ ${p.rating}</span>
            ${!p.in_stock ? '<span class="badge out-of-stock">Out of stock</span>' : ''}
          </div>
        </div>
      `).join('');
    }

    function selectProduct(id) {
      currentProductId = id;
      renderCatalog();
      const p = catalog.find(item => item.id === id);
      if (p) {
        document.getElementById('targetDisplay').innerHTML = `
          <div style="color: #f0f6fc; font-weight: 600; font-size: 1.05rem;">${p.title}</div>
          <div style="margin-top: 4px; font-size: 0.85rem; color: var(--text-muted);">${p.description}</div>
          <div style="margin-top: 8px; display: flex; gap: 8px;">
            <span class="badge cat">${p.category}</span>
            <span class="badge brand">${p.brand}</span>
            <span class="badge">Price: $${p.price.toFixed(2)}</span>
            <span class="badge rating">★ ${p.rating} (${p.reviews_count} reviews)</span>
          </div>
        `;
      }
      loadRecommendations();
    }

    async function loadRecommendations() {
      const inStock = document.getElementById('stockFilter').checked;
      const catCap = document.getElementById('categoryCap').value;

      const url = `/api/recommend/${currentProductId}?filter_out_of_stock=${inStock}&max_per_category=${catCap}&top_k=5`;
      const res = await fetch(url);
      const data = await res.json();

      document.getElementById('statRetrieval').textContent = data.stages_summary.stage_1_candidates_count;
      document.getElementById('statRanking').textContent = data.stages_summary.stage_2_ranked_count;
      document.getElementById('statReranking').textContent = data.stages_summary.stage_3_final_slate_count;

      const list = document.getElementById('recommendationsList');
      if (!data.recommendations.length) {
        list.innerHTML = '<p style="color: var(--text-muted);">No recommendations met criteria.</p>';
        return;
      }

      list.innerHTML = data.recommendations.map(r => `
        <div class="rec-card">
          <div class="rec-card-top">
            <div>
              <span class="rank-num">#${r.rank}</span>
              <strong style="color: #f0f6fc; font-size: 0.95rem;">${r.title}</strong>
            </div>
            <span style="font-weight: 600; color: #58a6ff;">$${r.price.toFixed(2)}</span>
          </div>
          <div style="font-size: 0.8rem; color: var(--text-muted); display:flex; gap: 8px;">
            <span class="badge cat">${r.category}</span>
            <span class="badge brand">${r.brand}</span>
            <span class="badge rating">★ ${r.rating}</span>
            <span class="badge" style="color: #bc8cff;">Channel: ${r.retrieval_source}</span>
          </div>
          <div class="score-breakdown">
            <div class="metric-pill">
              <span class="label">Total Rank Score</span>
              <span class="val" style="color: #3fb950;">${r.final_rank_score.toFixed(4)}</span>
            </div>
            <div class="metric-pill">
              <span class="label">Content Similarity</span>
              <span class="val">${r.content_similarity.toFixed(4)}</span>
            </div>
            <div class="metric-pill">
              <span class="label">Collaborative Score</span>
              <span class="val">${r.collaborative_score.toFixed(4)}</span>
            </div>
            <div class="metric-pill">
              <span class="label">Popularity Score</span>
              <span class="val">${r.popularity_score.toFixed(4)}</span>
            </div>
          </div>
          ${r.stage_notes && r.stage_notes.collab_reason ? `
            <div style="font-size: 0.75rem; color: #7ee787; background: #0c2d1c; padding: 4px 8px; border-radius: 4px;">
              💡 Social Proof: ${r.stage_notes.collab_reason}
            </div>
          ` : ''}
        </div>
      `).join('');
    }

    init();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/products")
def api_products():
    return jsonify(products_data)


@app.route("/api/recommend/<product_id>")
def api_recommend(product_id: str):
    top_k = int(request.args.get("top_k", 5))
    filter_stock = request.args.get("filter_out_of_stock", "true").lower() == "true"
    max_cat = int(request.args.get("max_per_category", 2))

    try:
        result = pipeline.recommend(
            target_product_id=product_id,
            top_k=top_k,
            filter_out_of_stock=filter_stock,
            max_per_category=max_cat,
        )
        return jsonify(result)
    except KeyError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/inspect-similarity")
def api_inspect_similarity():
    """Returns detailed similarity calculation between two products."""
    id1 = request.args.get("id1")
    id2 = request.args.get("id2")

    if not id1 or not id2:
        return jsonify({"error": "Both id1 and id2 parameters are required"}), 400

    if id1 not in content_engine.id_to_product or id2 not in content_engine.id_to_product:
        return jsonify({"error": "One or both product IDs not found"}), 404

    prod1 = content_engine.id_to_product[id1]
    prod2 = content_engine.id_to_product[id2]

    explanation = content_engine.compute_similarity(prod1, prod2)
    collab_sim, shared_users = collab_engine.item_similarity(id1, id2)

    return jsonify({
        "item1": {"id": prod1.id, "title": prod1.title, "tags": prod1.tags},
        "item2": {"id": prod2.id, "title": prod2.title, "tags": prod2.tags},
        "content_similarity": {
            "total_score": explanation.total_score,
            "text_similarity": explanation.text_similarity,
            "tag_similarity": explanation.tag_similarity,
            "category_match": explanation.category_match,
            "price_closeness": explanation.price_closeness,
        },
        "collaborative_similarity": {
            "score": round(collab_sim, 4),
            "shared_users_count": shared_users,
        },
    })


if __name__ == "__main__":
    print("Starting AI Recommendation System on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)

