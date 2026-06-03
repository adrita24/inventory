import threading
import uuid
import pytest
from db import init_db, get_conn, get_lock
from cart import ensure_session, add_to_cart, remove_from_cart, view_cart, clear_cart
from orders import place_order, get_order_history
from inventory import get_product, search_products, list_categories

def new_session():
    sid = str(uuid.uuid4())
    ensure_session(sid)
    return sid

def set_stock(pid, qty):
    conn = get_conn()
    lock = get_lock()
    with lock:
        conn.execute("UPDATE products SET quantity=?, version=0 WHERE product_id=?", (qty, pid))
        conn.commit()

def get_stock(pid):
    return get_product(pid)["quantity"]

ORIGINAL_STOCK = {
    "P007": 80,
    "P008": 45,
    "P009": 70,
    "P010": 100,
    "P016": 5,
}

def setup_module():
    init_db()

@pytest.fixture(autouse=True)
def restore_stock():
    yield
    for pid, qty in ORIGINAL_STOCK.items():
        set_stock(pid, qty)

def test_add_to_cart_basic():
    sid = new_session()
    r = add_to_cart(sid, "P007", 2)
    assert r["ok"]
    items = view_cart(sid)
    assert any(i["product_id"]=="P007" and i["quantity"]==2 for i in items)

def test_add_to_cart_accumulates():
    sid = new_session()
    add_to_cart(sid, "P007", 2)
    add_to_cart(sid, "P007", 3)
    items = view_cart(sid)
    match = next(i for i in items if i["product_id"]=="P007")
    assert match["quantity"] == 5

def test_add_to_cart_over_stock_rejected():
    sid = new_session()
    set_stock("P016", 3)
    r = add_to_cart(sid, "P016", 10)
    assert not r["ok"]
    assert "3" in r["reason"]

def test_add_to_cart_respects_existing_cart_quantity():
    sid = new_session()
    set_stock("P016", 3)
    add_to_cart(sid, "P016", 2)
    r = add_to_cart(sid, "P016", 2)
    assert not r["ok"]

def test_remove_from_cart():
    sid = new_session()
    add_to_cart(sid, "P007", 2)
    r = remove_from_cart(sid, "P007")
    assert r["ok"]
    assert view_cart(sid) == []

def test_remove_item_not_in_cart():
    sid = new_session()
    r = remove_from_cart(sid, "P007")
    assert not r["ok"]

def test_clear_cart():
    sid = new_session()
    add_to_cart(sid, "P007", 1)
    add_to_cart(sid, "P009", 1)
    clear_cart(sid)
    assert view_cart(sid) == []

def test_cart_subtotal_correct():
    sid = new_session()
    add_to_cart(sid, "P010", 3)
    items = view_cart(sid)
    match = next(i for i in items if i["product_id"]=="P010")
    assert match["subtotal"] == match["price"] * 3

def test_place_order_deducts_stock():
    sid = new_session()
    set_stock("P007", 10)
    before = get_stock("P007")
    add_to_cart(sid, "P007", 3)
    result = place_order(sid)
    assert result["ok"]
    assert get_stock("P007") == before - 3

def test_place_order_clears_cart():
    sid = new_session()
    set_stock("P007", 10)
    add_to_cart(sid, "P007", 1)
    place_order(sid)
    assert view_cart(sid) == []

def test_place_order_empty_cart():
    sid = new_session()
    r = place_order(sid)
    assert not r["ok"]
    assert "empty" in r["reason"].lower()

def test_place_order_insufficient_stock():
    sid = new_session()
    set_stock("P016", 2)
    add_to_cart(sid, "P016", 2)
    set_stock("P016", 1)
    r = place_order(sid)
    assert not r["ok"]
    assert "stock" in r["reason"].lower() or "insufficient" in r["reason"].lower()

def test_place_order_records_history():
    sid = new_session()
    set_stock("P009", 10)
    add_to_cart(sid, "P009", 2)
    place_order(sid)
    history = get_order_history(sid)
    assert any(o["name"] and o["quantity"]==2 for o in history)

def test_stock_never_goes_negative():
    sid = new_session()
    set_stock("P016", 1)
    add_to_cart(sid, "P016", 1)
    place_order(sid)
    assert get_stock("P016") == 0
    r = place_order(sid)
    assert not r["ok"]
    assert get_stock("P016") == 0

def test_no_oversell_under_concurrency():
    set_stock("P016", 5)
    wins = []
    errors = []

    def buyer():
        sid = new_session()
        add_to_cart(sid, "P016", 1)
        r = place_order(sid)
        if r["ok"]:
            wins.append(1)
        else:
            errors.append(r["reason"])

    threads = [threading.Thread(target=buyer) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(wins) == 5
    assert get_stock("P016") == 0

def test_concurrent_different_products():
    set_stock("P007", 5)
    set_stock("P009", 5)
    results = []

    def buyer(pid):
        sid = new_session()
        add_to_cart(sid, pid, 1)
        r = place_order(sid)
        results.append((pid, r["ok"]))

    threads = [
        threading.Thread(target=buyer, args=("P007" if i < 5 else "P009",))
        for i in range(10)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    p007_wins = sum(1 for pid, ok in results if pid=="P007" and ok)
    p009_wins = sum(1 for pid, ok in results if pid=="P009" and ok)
    assert p007_wins == 5
    assert p009_wins == 5

def test_exact_stock_boundary():
    set_stock("P016", 3)
    wins = []

    def buyer():
        sid = new_session()
        r = add_to_cart(sid, "P016", 1)
        if r["ok"]:
            r2 = place_order(sid)
            if r2["ok"]:
                wins.append(1)

    threads = [threading.Thread(target=buyer) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(wins) == 3
    assert get_stock("P016") == 0

def test_search_by_name():
    results = search_products("amul")
    assert any("Amul" in p["name"] for p in results)

def test_search_by_category():
    results = search_products(category="vegetables")
    assert all(p["category"].lower()=="vegetables" for p in results)
    assert len(results) > 0

def test_search_empty_query_returns_all():
    results = search_products("")
    assert len(results) > 0

def test_list_categories_not_empty():
    cats = list_categories()
    assert len(cats) > 0
    assert "Vegetables" in cats or "vegetables" in [c.lower() for c in cats]

def test_get_product_exists():
    p = get_product("P001")
    assert p is not None
    assert p["product_id"] == "P001"

def test_get_product_missing():
    p = get_product("ZZZZ")
    assert p is None
