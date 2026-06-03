from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from inventory import get_all_products

_vectorizer: TfidfVectorizer | None = None
_tfidf_matrix = None
_catalog: list[dict] = []

def build_index():
    global _vectorizer, _tfidf_matrix, _catalog
    _catalog = get_all_products()
    docs = [
        f"{p['name']} {p['category']} {p['description']}"
        for p in _catalog
    ]
    _vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    _tfidf_matrix = _vectorizer.fit_transform(docs)

def retrieve(query: str, top_k: int=5) -> list[dict]:
    if _vectorizer is None:
        build_index()
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [_catalog[i] for i in top_indices if scores[i] > 0]

def format_context(products: list[dict]) -> str:
    if not products:
        return "No matching products found in inventory."
    lines = ["Available products from inventory:"]
    for p in products:
        status = "In Stock" if p["quantity"] > 0 else "Out of Stock"
        lines.append(
            f"- [{p['product_id']}] {p['name']} | Rs{p['price']} | "
            f"Qty: {p['quantity']} ({status}) | {p['category']}"
        )
    return "\n".join(lines)
