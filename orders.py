from __future__ import annotations
import sqlite3
from datetime import datetime
from db import get_conn, get_lock
from cart import view_cart, clear_cart

def place_order(session_id: str) -> dict:
    conn = get_conn()
    lock = get_lock()

    cart_items = view_cart(session_id)
    if not cart_items:
        return {"ok": False, "reason": "Cart is empty."}

    with lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            placed = []
            order_total = 0.0
            now = datetime.utcnow().isoformat()

            for item in cart_items:
                pid = item["product_id"]
                qty = item["quantity"]

                fresh = conn.execute(
                    "SELECT name, price, quantity FROM products WHERE product_id = ?",
                    (pid,)
                ).fetchone()

                if fresh is None:
                    conn.execute("ROLLBACK")
                    return {"ok": False, "reason": f"Product {pid} no longer exists.", "failed_item": pid}

                if fresh["quantity"] < qty:
                    conn.execute("ROLLBACK")
                    return {
                        "ok": False,
                        "reason": (
                            f"Not enough stock for '{fresh['name']}': "
                            f"you want {qty}, only {fresh['quantity']} left."
                        ),
                        "failed_item": fresh["name"]
                    }

                conn.execute(
                    """UPDATE products
                       SET quantity=quantity-?, version=version+1, updated_at=?
                       WHERE product_id=?""",
                    (qty, now, pid)
                )

                line_total = qty * fresh["price"]
                conn.execute(
                    """INSERT INTO orders(session_id,product_id,quantity,price_each,total,placed_at)
                       VALUES (?,?,?,?,?,?)""",
                    (session_id, pid, qty, fresh["price"], line_total, now)
                )
                order_total += line_total
                placed.append({
                    "product_id": pid,
                    "name": fresh["name"],
                    "quantity": qty,
                    "price_each": fresh["price"],
                    "total": line_total
                })

            conn.execute("COMMIT")

        except sqlite3.OperationalError as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            return {"ok": False, "reason": f"Database error: {e}"}

    clear_cart(session_id)
    return {"ok": True, "orders": placed, "total": order_total}


def get_order_history(session_id: str) -> list[dict]:
    conn = get_conn()
    lock = get_lock()
    with lock:
        rows = conn.execute(
            """SELECT o.order_id, p.name, o.quantity, o.price_each, o.total, o.placed_at
               FROM orders o JOIN products p USING(product_id)
               WHERE o.session_id = ?
               ORDER BY o.placed_at DESC""",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]
