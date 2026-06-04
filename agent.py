from __future__ import annotations
import json
import os
import re
from groq import Groq

import inventory as inv
import cart as crt
import orders as ord_
import rag

_groq_client: Groq | None = None

def _groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return _groq_client

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

INTENT_PROMPT = """You are an intent classifier for a grocery chatbot. Extract the user's intent and return ONLY a JSON object, no explanation, no markdown.

Possible intents:
  search        - user wants to find/check a product or category. Extract "query" string.
  list_all      - user wants to see all available products.
  add_to_cart   - user wants to add a specific product. Extract "query" string and "quantity" integer.
  add_all       - user wants to add ALL products (or the whole inventory/everything) to cart. No query needed.
  remove_from_cart - user wants to remove a product. Extract "query" (product name, empty string if not mentioned) and "quantity" (integer or null -- null means remove all).
  clear_cart    - user wants to empty the cart entirely.
  view_cart     - user wants to see cart contents.
  place_order   - user wants to checkout / buy everything currently in cart.
  order_history - user wants to see past orders.
  unsupported   - user wants something the system cannot do, e.g. "buy all except X", partial checkout, apply coupon.
  chitchat      - anything else (greetings, random questions unrelated to shopping).

Rules:
- quantity defaults to 1 for add_to_cart if not specified.
- For remove_from_cart: if no product is named, set query to empty string "".
- For vague removes like "remove 2 units" or "remove them" with no product named, set query="" so the system resolves from cart context.
- "vegetables available" or "show dairy" -> intent: search, query: "vegetables" or "dairy".
- If the user says "yes" in reply to a suggestion, infer intent from conversation context.
- "add back", "no add back", "put it back", "add it again" -> intent: add_to_cart, query: "" (system resolves from context).
- "add everything", "add all products", "add the whole inventory", "add everything to cart" -> intent: add_all.
- "buy the whole inventory", "order everything", "buy everything" -> intent: add_all (user wants to add all then can place order separately; do NOT map to place_order).
- IMPORTANT: "buy all" with nothing else = place_order (cart already has items). "buy everything/whole inventory" = add_all (stocking the cart from scratch).

Return exactly this shape (no other text):
{"intent": "<intent>", "query": "<string>", "quantity": <integer or null>}

Examples:
"add 3 amul butter" -> {"intent": "add_to_cart", "query": "amul butter", "quantity": 3}
"remove tomatoes" -> {"intent": "remove_from_cart", "query": "tomatoes", "quantity": null}
"remove 2 units" -> {"intent": "remove_from_cart", "query": "", "quantity": 2}
"remove 1" -> {"intent": "remove_from_cart", "query": "", "quantity": 1}
"remove them" -> {"intent": "remove_from_cart", "query": "", "quantity": null}
"what's in my cart" -> {"intent": "view_cart", "query": "", "quantity": null}
"buy all" -> {"intent": "place_order", "query": "", "quantity": null}
"add everything to the cart" -> {"intent": "add_all", "query": "", "quantity": null}
"add all products" -> {"intent": "add_all", "query": "", "quantity": null}
"buy the whole inventory" -> {"intent": "add_all", "query": "", "quantity": null}
"order everything" -> {"intent": "add_all", "query": "", "quantity": null}
"buy everything" -> {"intent": "add_all", "query": "", "quantity": null}
"add the whole inventory to my cart" -> {"intent": "add_all", "query": "", "quantity": null}
"buy all except onions" -> {"intent": "unsupported", "query": "buy all except onions", "quantity": null}
"my budget is 500" -> {"intent": "unsupported", "query": "my budget is 500", "quantity": null}
"i have 1000 rupees" -> {"intent": "unsupported", "query": "i have 1000 rupees", "quantity": null}
"vegetables available" -> {"intent": "search", "query": "vegetables", "quantity": null}
"do you have coke" -> {"intent": "search", "query": "coke", "quantity": null}
"what products do you have" -> {"intent": "list_all", "query": "", "quantity": null}
"what do you sell" -> {"intent": "list_all", "query": "", "quantity": null}
"show me all products" -> {"intent": "list_all", "query": "", "quantity": null}
"what is available" -> {"intent": "list_all", "query": "", "quantity": null}
"how do 74 still remain" -> {"intent": "chitchat", "query": "how do 74 still remain", "quantity": null}
"why is the count still the same" -> {"intent": "chitchat", "query": "why is the count still the same", "quantity": null}
"no add back" -> {"intent": "add_to_cart", "query": "", "quantity": 1}
"add it back" -> {"intent": "add_to_cart", "query": "", "quantity": 1}
"put it back" -> {"intent": "add_to_cart", "query": "", "quantity": 1}
"add it again" -> {"intent": "add_to_cart", "query": "", "quantity": 1}
"""

RESPONSE_PROMPT = """You are Inventory tracker's shopping assistant. Write a short natural response based on the backend result.
- Format prices as Rs X.XX
- Do not invent any data not in the result.
- Be concise. 1-3 sentences max unless showing a list.
- If result says "Cart is empty." say exactly: Your cart is empty.
- Do not add suggestions or follow-up questions unless the result implies it.
"""

def _llm_call(system: str, user: str, max_tokens: int=256) -> str:
    resp = _groq().chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return resp.choices[0].message.content.strip()

def _extract_intent(message: str, history: list[dict]) -> dict:
    context = ""
    if history:
        recent = history[-4:]
        context = "\n".join(f"{h['role']}: {h['content']}" for h in recent)
        context = f"Recent conversation:\n{context}\n\n"
    raw = _llm_call(INTENT_PROMPT, f"{context}User message: {message}")
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "chitchat", "query": message, "quantity": None}

CATEGORIES = {"vegetables", "fruits", "dairy", "snacks", "beverages", "detergents", "household", "personal care"}

# ---------------------------------------------------------------------------
# Fuzzy helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _depluralise(word: str) -> str:
    """Very light stemmer: strip trailing 's' / 'es' so 'bananas'→'banana'."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and len(word) > 2:
        return word[:-1]
    return word


def _query_tokens(query: str) -> set[str]:
    stopwords = {"add", "remove", "get", "buy", "want", "give", "me", "a",
                 "an", "the", "some", "of", "to", "please", "i", "my"}
    tokens = _normalize(query).split()
    stems = set()
    for t in tokens:
        if t not in stopwords:
            stems.add(t)
            stems.add(_depluralise(t))
    return stems


def _product_tokens(product: dict) -> set[str]:
    blob = " ".join([
        product.get("name", ""),
        product.get("brand", ""),
        product.get("category", ""),
    ])
    tokens = _normalize(blob).split()
    stems = set()
    for t in tokens:
        stems.add(t)
        stems.add(_depluralise(t))
    return stems


def _fuzzy_token_match(query_stems: set[str], product_stems: set[str]) -> bool:
    """
    Return True when at least one query token is:
      - an exact match in the product tokens, OR
      - a prefix of a product token of length ≥ 4 (handles 'coca' in 'cocacola',
        'water' in 'watermelon'), but NOT short tokens like 'g' matching 'mango'.
    Both directions: query prefix of product, OR product prefix of query.
    """
    for qt in query_stems:
        if qt in product_stems:
            return True
        if len(qt) >= 4:
            for pt in product_stems:
                if len(pt) >= 4 and (pt.startswith(qt) or qt.startswith(pt)):
                    return True
    return False


def _resolve_product(query: str, session_id: str | None = None,
                     history: list[dict] | None = None,
                     use_cart_fallback: bool = True) -> dict | None:

    # 1. Explicit query — fuzzy token overlap
    if query.strip():
        hits = rag.retrieve(query, top_k=3)
        kw_hits = inv.search_products(query)
        seen_ids = {p["product_id"] for p in hits}
        for p in kw_hits:
            if p["product_id"] not in seen_ids:
                hits.append(p)

        query_stems = _query_tokens(query)

        if hits and query_stems:
            best = hits[0]
            product_stems = _product_tokens(best)
            if _fuzzy_token_match(query_stems, product_stems):
                return inv.get_product(best["product_id"])

        # RAG/keyword miss — fall back to a full-inventory fuzzy scan
        if query_stems:
            all_products = inv.get_all_products()
            best_match = None
            best_score = 0
            for p in all_products:
                p_stems = _product_tokens(p)
                score = 0
                for qt in query_stems:
                    if qt in p_stems:
                        score += 2  # exact match scores higher
                    elif len(qt) >= 4:
                        for pt in p_stems:
                            if len(pt) >= 4 and (pt.startswith(qt) or qt.startswith(pt)):
                                score += 1
                                break
                if score > best_score:
                    best_score = score
                    best_match = p
            # Require at least one real match (score > 0) to avoid false positives
            if best_match and best_score > 0:
                return inv.get_product(best_match["product_id"])

        return None  # genuinely not found

    # 2. Empty query — scan conversation history for a product mention
    #    (used for "add it back", "remove it", etc.)
    if history:
        all_products = inv.get_all_products()
        for turn in reversed(history[-6:]):
            content = turn.get("content", "").lower()
            for p in all_products:
                if p["name"].lower() in content:
                    return inv.get_product(p["product_id"])

    # 3. Cart fallback — only for remove/update contexts, never for add
    if use_cart_fallback and session_id:
        cart_items = crt.view_cart(session_id)
        if cart_items:
            return inv.get_product(cart_items[-1]["product_id"])

    return None


def _dispatch(intent: dict, session_id: str, history: list[dict] | None = None) -> str:
    action = intent.get("intent", "chitchat")
    query = (intent.get("query") or "").strip()
    raw_qty = intent.get("quantity")

    if action == "list_all":
        products = inv.get_all_products()
        if not products:
            return "Inventory is empty."
        lines = []
        current_cat = ""
        for p in products:
            if p["category"] != current_cat:
                current_cat = p["category"]
                lines.append(f"\n{current_cat}:")
            stock_label = "Out of Stock" if p["quantity"]==0 else f"{p['quantity']} in stock"
            lines.append(f"  [{p['product_id']}] {p['name']} - Rs{p['price']} ({stock_label})")
        return "\n".join(lines)

    elif action == "search":
        # Redirect generic "show all / list products / what do you have" queries
        _LIST_ALL_KEYWORDS = {"all", "everything", "products", "items", "inventory",
                              "available", "list", "show", "catalogue", "catalog"}
        if query and not (set(_normalize(query).split()) - _LIST_ALL_KEYWORDS):
            # query is entirely list-all keywords — treat as list_all
            return _dispatch({"intent": "list_all"}, session_id, history)

        if not query:
            cart_items = crt.view_cart(session_id)
            if cart_items:
                products = [inv.get_product(i["product_id"]) for i in cart_items]
                products = [p for p in products if p]
            else:
                return "What are you looking for?"
        elif query.lower() in CATEGORIES:
            products = inv.search_products(category=query)
        else:
            products = rag.retrieve(query, top_k=4)
            kw_hits = inv.search_products(query)
            seen_ids = {p["product_id"] for p in products}
            for p in kw_hits:
                if p["product_id"] not in seen_ids:
                    products.append(p)
            products = products[:5]
            # If nothing matched, say so clearly rather than asking "What are you looking for?"
            if not products:
                return f"Sorry, we don't carry '{query}'. Type 'show all' to see what's available."

        # Refresh every product from live DB — RAG catalog quantity is stale after orders
        fresh_products = []
        for p in products:
            fresh = inv.get_product(p["product_id"])
            if fresh:
                fresh_products.append(fresh)
        products = fresh_products

        cart_items = crt.view_cart(session_id)
        cart_map = {i["product_id"]: i["quantity"] for i in cart_items}
        adjusted = []
        for p in products:
            p = dict(p)
            p["quantity"] = p["quantity"] - cart_map.get(p["product_id"], 0)
            adjusted.append(p)
        if not adjusted:
            if query:
                return f"Sorry, we don't carry '{query}'. Type 'show all' to see what's available."
            return "What are you looking for?"
        return rag.format_context(adjusted)

    elif action == "add_to_cart":
        qty = raw_qty if raw_qty and raw_qty > 0 else 1
        # Allow history scan for empty queries ("add it back"), but never cart fallback for add
        product = _resolve_product(query, session_id, history, use_cart_fallback=False)
        if not product:
            if not query.strip():
                return "Which product do you want to add?"
            return f"Sorry, I couldn't find '{query}'. Try searching for what's available."
        result = crt.add_to_cart(session_id, product["product_id"], qty)
        if result["ok"]:
            fresh = inv.get_product(product["product_id"])
            db_stock = fresh["quantity"] if fresh else 0
            cart_items = crt.view_cart(session_id)
            in_cart = next((i["quantity"] for i in cart_items if i["product_id"]==product["product_id"]), 0)
            remaining = db_stock - in_cart
            return f"Added {qty}x {product['name']} to cart. ({remaining} units remaining in stock)"
        return f"Cannot add to cart: {result['reason']}"

    elif action == "add_all":
        products = inv.get_all_products()
        if not products:
            return "Inventory is empty, nothing to add."
        added, skipped = [], []
        for p in products:
            fresh = inv.get_product(p["product_id"])
            if not fresh or fresh["quantity"] == 0:
                skipped.append(p["name"])
                continue
            result = crt.add_to_cart(session_id, fresh["product_id"], 1)
            if result["ok"]:
                added.append(fresh["name"])
            else:
                skipped.append(fresh["name"])
        lines = []
        if added:
            lines.append(f"Added {len(added)} product(s) to cart: {', '.join(added)}.")
        if skipped:
            lines.append(f"Skipped (out of stock or error): {', '.join(skipped)}.")
        return " ".join(lines) if lines else "Nothing could be added."

    elif action == "remove_from_cart":
        product = _resolve_product(query, session_id, history)
        if not product:
            return "Could not figure out which product to remove. Can you name it?"
        pid = product["product_id"]
        cart_items = crt.view_cart(session_id)
        cart_entry = next((i for i in cart_items if i["product_id"]==pid), None)
        if not cart_entry:
            return f"{product['name']} is not in your cart."
        if raw_qty is None:
            result = crt.remove_from_cart(session_id, pid)
            return f"Removed {product['name']} from cart." if result["ok"] else result["reason"]
        new_qty = cart_entry["quantity"] - raw_qty
        if new_qty <= 0:
            result = crt.remove_from_cart(session_id, pid)
            return f"Removed {product['name']} from cart." if result["ok"] else result["reason"]
        crt.remove_from_cart(session_id, pid)
        crt.add_to_cart(session_id, pid, new_qty)
        return f"Updated {product['name']} in cart: {new_qty} units."

    elif action == "clear_cart":
        crt.clear_cart(session_id)
        return "Cart cleared."

    elif action == "view_cart":
        items = crt.view_cart(session_id)
        if not items:
            return "Cart is empty."
        lines = ["Cart contents:"]
        grand_total = 0.0
        for item in items:
            lines.append(f"  [{item['product_id']}] {item['name']} x{item['quantity']} @ Rs{item['price']} = Rs{item['subtotal']:.2f}")
            grand_total += item["subtotal"]
        lines.append(f"Grand Total: Rs{grand_total:.2f}")
        return "\n".join(lines)

    elif action == "place_order":
        result = ord_.place_order(session_id)
        if result["ok"]:
            lines = ["Order placed!"]
            for o in result["orders"]:
                lines.append(f"  {o['name']} x{o['quantity']} = Rs{o['total']:.2f}")
            lines.append(f"Total: Rs{result['total']:.2f}")
            return "\n".join(lines)
        return f"Order failed: {result['reason']}"

    elif action == "order_history":
        history_ = ord_.get_order_history(session_id)
        if not history_:
            return "No orders placed yet."
        lines = ["Order history:"]
        for o in history_:
            lines.append(f"  #{o['order_id']}: {o['name']} x{o['quantity']} = Rs{o['total']:.2f} on {o['placed_at'][:10]}")
        return "\n".join(lines)

    elif action == "unsupported":
        return "I can't do that. I support searching products, managing your cart, and placing orders. No budgets, coupons, or partial checkouts."

    return "CHITCHAT"

_DIRECT_INTENTS = {
    "add_to_cart", "remove_from_cart", "clear_cart",
    "view_cart", "place_order", "order_history", "unsupported",
    "search", "list_all", "add_all",
}

def chat(session_id: str, message: str, history: list[dict]) -> str:
    intent = _extract_intent(message, history)
    action = intent.get("intent", "chitchat")
    backend_result = _dispatch(intent, session_id, history)

    if backend_result == "CHITCHAT":
        ctx = ""
        if history:
            recent = history[-4:]
            ctx = "\nRecent conversation:\n" + "\n".join(f"{h['role']}: {h['content']}" for h in recent)
        return _llm_call(
            "You are Inventory tracker's shopping assistant. Answer in context of the conversation. Keep it brief.",
            f"{ctx}\n\nUser: {message}",
            max_tokens=200
        )

    if action in _DIRECT_INTENTS:
        return backend_result

    return _llm_call(RESPONSE_PROMPT, f"Backend result:\n{backend_result}", max_tokens=512)