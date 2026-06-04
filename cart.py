from __future__ import annotations
from datetime import datetime
from db import get_conn, get_lock


def ensure_session(sid: str):
    conn = get_conn()
    lock = get_lock()
    with lock:
        conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, created_at) VALUES (?,?)",
            (sid, datetime.utcnow().isoformat())
        )
        conn.commit()


def add_to_cart(sid: str, pid: str, qty: int) -> dict:
    if qty <= 0:
        return {"ok": False, "reason": "Quantity must be a positive integer."}
    conn = get_conn()
    lock = get_lock()
    with lock:
        product = conn.execute(
            "SELECT name, quantity FROM products WHERE product_id = ?", (pid,)
        ).fetchone()
        if not product:
            return {"ok": False, "reason": f"Product {pid} not found."}

        existing = conn.execute(
            "SELECT quantity FROM cart WHERE session_id=? AND product_id=?",
            (sid, pid)
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
                (qty, sid, pid)
            )
        else:
            conn.execute(
                "INSERT INTO cart(session_id, product_id, quantity) VALUES (?,?,?)",
                (sid, pid, qty)
            )
        conn.commit()
    return {"ok": True, "name": product["name"], "qty": already_in_cart + qty}


def remove_from_cart(sid: str, pid: str) -> dict:
    conn = get_conn()
    lock = get_lock()
    with lock:
        row = conn.execute(
            "SELECT product_id FROM cart WHERE session_id=? AND product_id=?",
            (sid, pid)
        ).fetchone()
        if not row:
            return {"ok": False, "reason": "Item not in cart."}
        conn.execute(
            "DELETE FROM cart WHERE session_id=? AND product_id=?",
            (sid, pid)
        )
        conn.commit()
    return {"ok": True}


def reduce_cart_quantity(sid: str, pid: str, by: int) -> dict:
    if by <= 0:
        return {"ok": False, "reason": "Reduction amount must be positive."}
    conn = get_conn()
    lock = get_lock()
    with lock:
        row = conn.execute(
            "SELECT quantity FROM cart WHERE session_id=? AND product_id=?",
            (sid, pid)
        ).fetchone()
        if not row:
            return {"ok": False, "reason": "Item not in cart."}
        new_qty = row["quantity"] - by
        if new_qty <= 0:
            conn.execute(
                "DELETE FROM cart WHERE session_id=? AND product_id=?",
                (sid, pid)
            )
            conn.commit()
            return {"ok": True, "new_qty": 0}
        conn.execute(
            "UPDATE cart SET quantity=? WHERE session_id=? AND product_id=?",
            (new_qty, sid, pid)
        )
        conn.commit()
    return {"ok": True, "new_qty": new_qty}


def view_cart(sid: str) -> list[dict]:
    conn = get_conn()
    lock = get_lock()
    with lock:
        rows = conn.execute(
            """SELECT c.product_id, p.name, p.category, p.price, c.quantity,
                      c.quantity * p.price AS subtotal
               FROM cart c JOIN products p USING(product_id)
               WHERE c.session_id = ?""",
            (sid,)
        ).fetchall()
    return [dict(r) for r in rows]


def clear_cart(sid: str):
    conn = get_conn()
    lock = get_lock()
    with lock:
        conn.execute("DELETE FROM cart WHERE session_id=?", (sid,))
        conn.commit()