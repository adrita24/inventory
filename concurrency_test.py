import threading
import sqlite3
import time
import uuid
from datetime import datetime, timezone

from db import init_db, get_conn, get_lock, DB_PATH
from cart import ensure_session, add_to_cart
from orders import place_order

# tweak these before each demo run
NUM_USERS       = 6
STOCK_QTY       = 5
TARGET_QTY_EACH = 1
PRODUCT_ID      = "P016"

pessimistic_results: list[dict] = []
optimistic_results:  list[dict] = []

def now():
    return datetime.now(timezone.utc).isoformat()

def reset_stock(qty: int):
    conn = get_conn()
    lock = get_lock()
    with lock:
        conn.execute(
            "UPDATE products SET quantity=?, version=0, updated_at=? WHERE product_id=?",
            (qty, now(), PRODUCT_ID)
        )
        conn.commit()

def pessimistic_buyer(user_num: int, barrier: threading.Barrier):
    sid = str(uuid.uuid4())
    ensure_session(sid)
    add_to_cart(sid, PRODUCT_ID, TARGET_QTY_EACH)
    barrier.wait()
    t0 = time.perf_counter()
    result = place_order(sid)
    elapsed = time.perf_counter() - t0
    pessimistic_results.append({
        "user": f"user-{user_num:02d}",
        "ok": result["ok"],
        "reason": result.get("reason", ""),
        "ms": round(elapsed * 1000),
    })

def optimistic_buyer(user_num: int, barrier: threading.Barrier, max_retries: int=5):
    barrier.wait()
    t0 = time.perf_counter()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=3000")
    success = False
    for attempt in range(max_retries):
        row = conn.execute(
            "SELECT quantity, version FROM products WHERE product_id=?", (PRODUCT_ID,)
        ).fetchone()
        if row is None or row["quantity"] < TARGET_QTY_EACH:
            break
        ver = row["version"]
        cur = conn.execute(
            """UPDATE products SET quantity=quantity-?, version=version+1, updated_at=?
               WHERE product_id=? AND version=? AND quantity>=?""",
            (TARGET_QTY_EACH, now(), PRODUCT_ID, ver, TARGET_QTY_EACH)
        )
        conn.commit()
        if cur.rowcount == 1:
            success = True
            break
        time.sleep(0.002 * (2 ** attempt))
    conn.close()
    elapsed = time.perf_counter() - t0
    optimistic_results.append({
        "user": f"user-{user_num:02d}",
        "ok": success,
        "reason": "" if success else "lost the CAS race, retries exhausted",
        "ms": round(elapsed * 1000),
    })

def run_scenario(label: str, buyer_fn, results: list):
    results.clear()
    reset_stock(STOCK_QTY)
    max_winners = min(NUM_USERS, STOCK_QTY // TARGET_QTY_EACH)

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  stock={STOCK_QTY}  users={NUM_USERS}  each wants={TARGET_QTY_EACH}  max can win={max_winners}")
    print(f"{'─'*60}")

    gate = threading.Barrier(NUM_USERS)
    threads = [
        threading.Thread(target=buyer_fn, args=(i, gate))
        for i in range(1, NUM_USERS+1)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    for r in sorted(results, key=lambda x: x["user"]):
        tag  = "ok  " if r["ok"] else "fail"
        note = f"  ({r['reason']})" if r["reason"] else ""
        print(f"  {tag}  {r['user']}  {r['ms']}ms{note}")

    conn = get_conn()
    lock = get_lock()
    with lock:
        left = conn.execute(
            "SELECT quantity FROM products WHERE product_id=?", (PRODUCT_ID,)
        ).fetchone()["quantity"]

    wins = sum(1 for r in results if r["ok"])
    sold = STOCK_QTY - left

    print(f"\n  won={wins}  lost={NUM_USERS-wins}  stock left={left}  units sold={sold}")

    assert wins == sold, f"mismatch: {wins} wins but {sold} units deducted"
    assert left >= 0,    "negative stock — oversold!"
    print(f"  inventory check passed.")

if __name__ == "__main__":
    print("setting up db...")
    init_db()

    run_scenario(
        "pessimistic locking  (BEGIN IMMEDIATE)",
        pessimistic_buyer,
        pessimistic_results,
    )
    run_scenario(
        "optimistic locking  (CAS on version column)",
        optimistic_buyer,
        optimistic_results,
    )

    print(f"\n{'─'*60}")
    print("  summary")
    print(f"{'─'*60}")
    print("""
  pessimistic (BEGIN IMMEDIATE)
    good:  all available stock gets sold, no spurious failures
    good:  simple, no retry logic
    tradeoff:  writers queue up — you can see the latency above

  optimistic (CAS on version)
    good:  no blocking on reads, lower latency at low contention
    tradeoff:  under high contention some buyers fail even with
               stock available — retry budget runs out
    tradeoff:  more complex, need retry logic

  for a checkout flow, pessimistic is the right default.
  optimistic makes sense for things like view counters or likes.
""")
    print("all checks passed.")
