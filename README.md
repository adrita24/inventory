# Inventory Management RAG Chatbot

A production-grade inventory management system with a conversational interface, built for correctness under concurrent load. The chatbot is an interface to a real backend -- all business logic, stock management, and transaction handling is implemented in Python and SQLite, not in the LLM.

---

## Architecture

```
User (Browser)
    |
Streamlit (app.py)
    |
Intent Classifier (agent.py)
    |-- LLM classifies message into structured intent JSON
    |-- Python dispatches to backend based on intent
    |-- For transactional results: returned verbatim (no LLM paraphrasing)
    |-- For search/list: LLM formats for readability
    |
Backend Functions
    |-- inventory.py   read-only product queries
    |-- cart.py        session-scoped cart CRUD
    |-- orders.py      checkout with pessimistic locking
    |-- rag.py         TF-IDF retrieval for product search
    |
SQLite (inventory.db)
    WAL mode, BEGIN IMMEDIATE transactions, busy_timeout=5000ms
```

### Why not ReAct / tool-calling?

ReAct gives the LLM autonomy over when to call tools. Under long context, the model begins answering from memory instead of calling the tool, generating partial structured output that fails mid-generation (HTTP 400). For a system with real transactional state, this is unacceptable.

The two-step approach used here separates concerns cleanly:
- LLM does one thing: classify intent into a small JSON blob.
- Python does everything else: resolve products, read/write DB, format output.

The LLM never touches numerical data (prices, quantities, stock levels) in the response path for transactional intents. This eliminates hallucinated prices entirely.

---

## Concurrency Design

### The Problem

Without locking, concurrent checkout creates a TOCTOU race:
1. User A reads quantity = 5, passes stock check
2. User B reads quantity = 5, passes stock check
3. User A writes quantity = 4
4. User B writes quantity = 4
Result: 2 units sold, 1 unit deducted. Inventory corrupted.

### Pessimistic Locking (production implementation)

Location: `orders.place_order()`

```
threading.Lock acquired (serialises Streamlit threads)
    BEGIN IMMEDIATE (SQLite write lock acquired at transaction start)
        SELECT quantity -- fresh read, no stale cache possible
        if quantity < requested: ROLLBACK, return error
        UPDATE quantity = quantity - N, version = version + 1
        INSERT INTO orders
    COMMIT
Lock released
```

`BEGIN IMMEDIATE` vs `BEGIN DEFERRED`: DEFERRED acquires the write lock only on first write. The window between the stock read and the write is a race condition. IMMEDIATE closes that window by locking on BEGIN.

`busy_timeout=5000`: concurrent writers queue and retry for up to 5 seconds instead of failing immediately.

`threading.Lock`: SQLite's locking is at the process level. Within a single Streamlit process running multiple user sessions as threads, the Python lock serialises access before hitting SQLite.

### Optimistic Locking (demonstrated in concurrency_test.py)

Each row has a `version INTEGER` column. Readers read `(quantity, version)`, writers do:

```sql
UPDATE products
SET quantity = quantity - N, version = version + 1
WHERE product_id = ? AND version = ? AND quantity >= N
```

If `rowcount == 0`, another writer committed between the read and this write. Retry with exponential backoff. Under high contention, retries exhaust and the user fails even when stock is available -- a spurious failure. This is acceptable for low-write workloads (e.g. view counters) but not for checkout.

### Isolation Level

SQLite default is SERIALIZABLE. With WAL mode: readers never block writers, writers never block readers. Read throughput is preserved while writes are serialised.

### Distributed Scalability

This implementation uses SQLite and is single-machine. For multi-process or multi-machine deployment:

- Replace SQLite with PostgreSQL
- Use `SELECT ... FOR UPDATE SKIP LOCKED` for row-level pessimistic locking
- Use Redis `SET NX` or Lua scripts for distributed locks across nodes
- Use a message queue (RabbitMQ, Kafka) to serialise order events and decouple checkout from inventory

---

## Database Schema

```sql
CREATE TABLE products (
    product_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    description TEXT NOT NULL,
    price       REAL NOT NULL,
    quantity    INTEGER NOT NULL CHECK(quantity >= 0),
    version     INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL
);

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE cart (
    cart_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    quantity   INTEGER NOT NULL CHECK(quantity > 0),
    UNIQUE(session_id, product_id)
);

CREATE TABLE orders (
    order_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity   INTEGER NOT NULL,
    price_each REAL NOT NULL,
    total      REAL NOT NULL,
    placed_at  TEXT NOT NULL
);
```

The `version` column on `products` is the optimistic lock counter. Every write increments it. A CAS update that finds a different version fails and retries.

---

## Intent Classification

User messages are classified into one of these intents before any backend call is made:

| Intent | Trigger examples | Backend call |
|---|---|---|
| search | "do you have detergent", "show dairy" | TF-IDF retrieval + keyword search |
| list_all | "what products do you have" | Full catalog query |
| stock_check | "how many X remain", "X quantity" | Direct product lookup |
| add_to_cart | "add 3 amul butter" | cart.add_to_cart |
| remove_from_cart | "remove tomatoes", "remove 2 units" | cart.remove_from_cart |
| clear_cart | "clear my cart", "empty cart" | cart.clear_cart |
| view_cart | "what's in my cart" | cart.view_cart |
| place_order | "buy all", "checkout" | orders.place_order (locked) |
| order_history | "show my orders" | orders.get_order_history |
| unsupported | "buy all except X" | Rejected with explanation |
| chitchat | anything else | LLM response only |

For all transactional intents (add, remove, clear, view, place, history), the backend result is returned verbatim to the user. The LLM is not in the response path for these -- it cannot hallucinate prices or quantities.

For vague removes ("remove 2 units" with no product named), the system resolves the product from the last item in the session cart.

---

## File Structure

```
inventory_rag_chatbot/
    app.py                  Streamlit frontend, session management, sidebar cart view
    agent.py                Intent classifier + deterministic dispatch
    rag.py                  TF-IDF vector store (sklearn), product retrieval
    db.py                   SQLite connection, WAL config, schema, sample data
    inventory.py            Read-only product queries
    cart.py                 Session-scoped cart CRUD
    orders.py               Checkout with BEGIN IMMEDIATE transaction
    concurrency_test.py     Race condition demonstration, both locking strategies
    requirements.txt        Python dependencies
    inventory.db            Auto-created on first run, do not commit
```

---

## Setup

```
pip install -r requirements.txt
```

Set the Groq API key:
```
export GROQ_API_KEY=your_key_here      # Linux/Mac
set GROQ_API_KEY=your_key_here         # Windows CMD
$env:GROQ_API_KEY="your_key_here"      # Windows PowerShell
```

Or add it to a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

Run the app:
```
streamlit run app.py
```

Run the concurrency test:
```
python concurrency_test.py
```

---

## Concurrency Test

The test script demonstrates race condition prevention with configurable parameters at the top of the file:

```python
NUM_USERS       = 6    # concurrent users attempting purchase
STOCK_QTY       = 5    # units in stock before each scenario
TARGET_QTY_EACH = 1    # units each user wants to buy
PRODUCT_ID      = "P016"
```

Change these values and re-run to demonstrate different scenarios live. The invariant checked after every run:

```
orders recorded == units deducted from stock
remaining stock >= 0
```

If either invariant breaks, the script raises AssertionError. It has not.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq LLaMA 4 Scout 17B |
| Vector retrieval | TF-IDF + cosine similarity (scikit-learn) |
| Database | SQLite with WAL mode |
| Frontend | Streamlit |
| Concurrency | threading.Lock + BEGIN IMMEDIATE + optimistic CAS |
| Language | Python 3.10+ |
