from __future__ import annotations
from datetime import datetime
from db import get_conn, get_lock

def ensure_session(session_id: str):
    conn = get_conn()
    lock = get_lock()
    with lock:
        conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, created_at) VALUES (?,?)",
            (session_id, datetime.utcnow().isoformat())
        )
        conn.commit()

def add_to_cart(session_id: str, product_id: str, qty: int) -> dict:
    conn = get_conn()
    lock = get_lock()
    with lock:
        product = conn.execute(
            "SELECT name, quantity FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
        if not product:
            return {"ok": False, "reason": f"Product {product_id} not found."}

        existing = conn.execute(
            "SELECT quantity FROM cart WHERE session_id=? AND product_id=?",
            (session_id, product_id)
        ).fetchone()
        already_in_cart = existing["quantity"] if existing else 0

        if already_in_cart + qty > product["quantity"]:
            return {
                "ok": False,
                "reason": f"Only {product['quantity']} units of '{product['name']}' available "
                           f"(you already have {already_in_cart} in cart)."
            }

        if existing:
            conn.execute(
                "UPDATE cart SET quantity=quantity+? WHERE session_id=? AND product_id=?",
                (qty, session_id, product_id)
            )
        else:
            conn.execute(
                "INSERT INTO cart(session_id, product_id, quantity) VALUES (?,?,?)",
                (session_id, product_id, qty)
            )
        conn.commit()
    return {"ok": True, "name": product["name"], "qty": already_in_cart + qty}

def remove_from_cart(session_id: str, product_id: str) -> dict:
    conn = get_conn()
    lock = get_lock()
    with lock:
        row = conn.execute(
            "SELECT product_id FROM cart WHERE session_id=? AND product_id=?",
            (session_id, product_id)
        ).fetchone()
        if not row:
            return {"ok": False, "reason": "Item not in cart."}
        conn.execute(
            "DELETE FROM cart WHERE session_id=? AND product_id=?",
            (session_id, product_id)
        )
        conn.commit()
    return {"ok": True}

def view_cart(session_id: str) -> list[dict]:
    conn = get_conn()
    lock = get_lock()
    with lock:
        rows = conn.execute(
            """SELECT c.product_id, p.name, p.price, c.quantity,
                      c.quantity * p.price AS subtotal
               FROM cart c JOIN products p USING(product_id)
               WHERE c.session_id = ?""",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]

def clear_cart(session_id: str):
    conn = get_conn()
    lock = get_lock()
    with lock:
        conn.execute("DELETE FROM cart WHERE session_id=?", (session_id,))
        conn.commit()
