from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from inventory import get_all_products

_vectorizer: TfidfVectorizer | None = None
_tfidf_matrix = None
_catalog: list[dict] = []
_catalog_size: int = 0


def build_index():
    global _vectorizer, _tfidf_matrix, _catalog, _catalog_size
    _catalog = get_all_products()
    _catalog_size = len(_catalog)
    docs = [
        f"{p['name']} {p['category']} {p['description']}"
        for p in _catalog
    ]
    _vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    _tfidf_matrix = _vectorizer.fit_transform(docs)


def _ensure_index():
    if _vectorizer is None:
        build_index()
        return
    current = get_all_products()
    if len(current) != _catalog_size:
        build_index()


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    _ensure_index()
    q_vec = _vectorizer.transform([query])
    scores = cosine_similarity(q_vec, _tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [_catalog[i] for i in top_idx if scores[i] > 0]


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