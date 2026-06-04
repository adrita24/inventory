import sqlite3
import threading
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "inventory.db")

_thread_local = threading.local()
_global_lock  = threading.Lock()


def get_conn() -> sqlite3.Connection:
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        _thread_local.conn = conn
    return conn


def get_lock() -> threading.Lock:
    return _global_lock


SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    description  TEXT NOT NULL,
    price        REAL NOT NULL,
    quantity     INTEGER NOT NULL CHECK(quantity >= 0),
    version      INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cart (
    cart_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    quantity   INTEGER NOT NULL CHECK(quantity > 0),
    UNIQUE(session_id, product_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity   INTEGER NOT NULL,
    price_each REAL NOT NULL,
    total      REAL NOT NULL,
    placed_at  TEXT NOT NULL
);
"""

SAMPLE_DATA = [
    ("P001", "Amul Full Cream Milk 1L",    "Dairy",         "Fresh full cream milk, 1 litre pack",               68.0,  50),
    ("P002", "Amul Butter 500g",           "Dairy",         "Pasteurised table butter, salted, 500g",            295.0, 30),
    ("P003", "Nestle Dahi 400g",           "Dairy",         "Creamy set curd, 400g cup",                         55.0,  40),
    ("P004", "Bananas 1 dozen",            "Fruits",        "Fresh yellow bananas, approx 12 pieces",            49.0,  60),
    ("P005", "Royal Gala Apples 4 pcs",    "Fruits",        "Crisp imported apples, pack of 4",                  149.0, 25),
    ("P006", "Watermelon 1 pc",            "Fruits",        "Sweet seedless watermelon, ~2kg",                   89.0,  15),
    ("P007", "Tomatoes 500g",              "Vegetables",    "Fresh red tomatoes, 500g pack",                     35.0,  80),
    ("P008", "Spinach 250g",               "Vegetables",    "Farm-fresh palak, 250g bundle",                     25.0,  45),
    ("P009", "Onions 1kg",                 "Vegetables",    "Red onions, 1kg loose pack",                        42.0,  70),
    ("P010", "Lay's Classic Salted 52g",   "Snacks",        "Classic salted potato chips, 52g",                  20.0,  100),
    ("P011", "Bingo Mad Angles 90g",       "Snacks",        "Achaari masti flavour chips, 90g",                  30.0,  80),
    ("P012", "Parle-G 800g",               "Snacks",        "Original glucose biscuits, family pack 800g",       60.0,  60),
    ("P013", "Coca-Cola 2L",               "Beverages",     "Chilled sparkling cola, 2 litre bottle",            95.0,  40),
    ("P014", "Red Bull 250ml",             "Beverages",     "Energy drink, original, 250ml can",                 125.0, 30),
    ("P015", "Tropicana Orange 1L",        "Beverages",     "100% fruit juice, no added sugar, 1L",              130.0, 35),
    ("P016", "Surf Excel Easy Wash 1kg",   "Detergents",    "Stain-removing detergent powder, 1kg",              175.0, 5),
    ("P017", "Vim Dishwash Bar 400g",      "Detergents",    "Lemon dishwash bar for utensils, 400g",             55.0,  50),
    ("P018", "Harpic Power Plus 500ml",    "Household",     "Toilet cleaner, max formula, 500ml",                115.0, 40),
    ("P019", "Colgate MaxFresh 150g",      "Personal Care", "Cooling crystals toothpaste, 150g",                 95.0,  55),
    ("P020", "Dove Body Lotion 200ml",     "Personal Care", "Moisturising body lotion, 200ml",                   265.0, 30),
]


def init_db():
    conn = get_conn()
    with _global_lock:
        conn.executescript(SCHEMA)
        ts = datetime.utcnow().isoformat()
        for row in SAMPLE_DATA:
            conn.execute(
                """INSERT OR IGNORE INTO products
                   (product_id,name,category,description,price,quantity,updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (*row, ts)
            )
        conn.commit()