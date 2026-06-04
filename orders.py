from __future__ import annotations
import sqlite3
from datetime import datetime
from db import get_conn, get_lock
from cart import clear_cart


def place_order(sid: str) -> dict:
    conn = get_conn()
    lock = get_lock()

    with lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cart_rows = conn.execute(
                """SELECT c.product_id, c.quantity
                   FROM cart c
                   WHERE c.session_id = ?""",
                (sid,)
            ).fetchall()

            if not cart_rows:
                conn.execute("ROLLBACK")
                return {"ok": False, "reason": "Cart is empty."}

            placed = []
            running_total = 0.0
            ts = datetime.utcnow().isoformat()

            for row in cart_rows:
                pid = row["product_id"]
                qty = row["quantity"]

                product = conn.execute(
                    "SELECT name, price, quantity FROM products WHERE product_id = ?",
                    (pid,)
                ).fetchone()

                if product is None:
                    conn.execute("ROLLBACK")
                    return {"ok": False, "reason": f"Product {pid} no longer exists.", "failed_item": pid}

                if product["quantity"] < qty:
                    conn.execute("ROLLBACK")
                    return {
                        "ok": False,
                        "reason": (
                            f"Not enough stock for '{product['name']}': "
                            f"you want {qty}, only {product['quantity']} left."
                        ),
                        "failed_item": product["name"],
                    }

                conn.execute(
                    """UPDATE products
                       SET quantity=quantity-?, version=version+1, updated_at=?
                       WHERE product_id=?""",
                    (qty, ts, pid)
                )

                line_total = qty * product["price"]
                conn.execute(
                    """INSERT INTO orders(session_id,product_id,quantity,price_each,total,placed_at)
                       VALUES (?,?,?,?,?,?)""",
                    (sid, pid, qty, product["price"], line_total, ts)
                )
                running_total += line_total
                placed.append({
                    "product_id": pid,
                    "name":       product["name"],
                    "quantity":   qty,
                    "price_each": product["price"],
                    "total":      line_total,
                })

            conn.execute("COMMIT")

        except sqlite3.OperationalError as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            return {"ok": False, "reason": f"Database error: {e}"}

    clear_cart(sid)
    return {"ok": True, "orders": placed, "total": running_total}


def get_order_history(sid: str) -> list[dict]:
    conn = get_conn()
    lock = get_lock()
    with lock:
        rows = conn.execute(
            """SELECT o.order_id, p.name, o.quantity, o.price_each, o.total, o.placed_at
               FROM orders o JOIN products p USING(product_id)
               WHERE o.session_id = ?
               ORDER BY o.placed_at DESC""",
            (sid,)
        ).fetchall()
    return [dict(r) for r in rows]