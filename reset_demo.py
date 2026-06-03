import sys
from datetime import datetime, timezone
from db import init_db, get_conn, get_lock

pid = sys.argv[1] if len(sys.argv) > 1 else "P016"
qty = int(sys.argv[2]) if len(sys.argv) > 2 else 5

init_db()
conn = get_conn()
lock = get_lock()

with lock:
    row = conn.execute("SELECT name, quantity FROM products WHERE product_id=?", (pid,)).fetchone()
    if not row:
        print(f"product {pid} not found")
        sys.exit(1)
    conn.execute(
        "UPDATE products SET quantity=?, version=0, updated_at=? WHERE product_id=?",
        (qty, datetime.now(timezone.utc).isoformat(), pid)
    )
    conn.commit()

print(f"[{pid}] {row['name']}: {row['quantity']} -> {qty} units, version reset to 0")
