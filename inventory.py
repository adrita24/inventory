from __future__ import annotations
from db import get_conn, get_lock

def search_products(query: str="", category: str="") -> list[dict]:
    conn = get_conn()
    lock = get_lock()
    pattern = f"%{query.lower()}%"
    sql = """
        SELECT product_id, name, category, description, price, quantity, updated_at
        FROM products
        WHERE (LOWER(name) LIKE ? OR LOWER(description) LIKE ?)
    """
    params: list = [pattern, pattern]
    if category:
        sql += " AND LOWER(category) = ?"
        params.append(category.lower())
    sql += " ORDER BY category, name"
    with lock:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]

def get_product(product_id: str) -> dict | None:
    conn = get_conn()
    lock = get_lock()
    with lock:
        row = conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
    return dict(row) if row else None

def list_categories() -> list[str]:
    conn = get_conn()
    lock = get_lock()
    with lock:
        rows = conn.execute(
            "SELECT DISTINCT category FROM products ORDER BY category"
        ).fetchall()
    return [r[0] for r in rows]

def get_all_products() -> list[dict]:
    conn = get_conn()
    lock = get_lock()
    with lock:
        rows = conn.execute(
            "SELECT product_id, name, category, description, price, quantity FROM products ORDER BY category, name"
        ).fetchall()
    return [dict(r) for r in rows]
