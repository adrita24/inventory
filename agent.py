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

SYNONYMS: dict[str, str] = {
    "coke":         "coca cola",
    "cocacola":     "coca cola",
    "redbull":      "red bull",
    "oj":           "orange juice",
    "orange juice": "tropicana orange",
    "fruit juice":  "tropicana",
    "yogurt":       "dahi",
    "yoghurt":      "dahi",
    "curd":         "dahi",
    "ghee":         "butter",
    "chips":        "salted",
    "biscuits":     "parle",
    "biscuit":      "parle",
    "cookies":      "parle",
    "glucose":      "parle",
    "soap":         "dishwash",
    "dishsoap":     "dishwash",
    "washing powder": "surf excel",
    "washing":      "surf",
    "laundry":      "surf",
    "moisturiser":  "lotion",
    "moisturizer":  "lotion",
    "body cream":   "lotion",
    "toothpaste":   "colgate",
    "paste":        "colgate",
    "tomato":       "tomatoes",
    "onion":        "onions",
    "palak":        "spinach",
    "banana":       "bananas",
    "apple":        "apples",
}

INTENT_PROMPT = """You are an intent classifier for a grocery chatbot. Extract the user's intent and return ONLY a JSON object, no explanation, no markdown.

Possible intents:
  search           - user wants to find/check a product or category.
  list_all         - user wants to see all available products.
  add_to_cart      - user wants to add one or more SPECIFIC products to the cart.
  add_all          - user wants to add ALL products (the whole inventory) to the cart, with an optional per-product quantity.
  add_category     - user wants to add ALL products within a SPECIFIC category (e.g. all detergents, all dairy), with an optional quantity.
  remove_from_cart - user wants to remove a specific product from the cart.
  reduce_all       - user wants to reduce ALL cart items by the same quantity, or remove N of everything.
  clear_cart       - user wants to empty the cart entirely (ALL items gone).
  view_cart        - user wants to see what is currently in their cart.
  place_order      - user wants to checkout / buy everything currently in the cart.
  order_history    - user wants to see past orders.
  unsupported      - user wants something the system cannot do (budgets, coupons, partial checkout, admin/inventory edits, adding items to the store catalogue, etc.).
  chitchat         - anything else unrelated to shopping.

Rules:
- For add_to_cart, ALWAYS return "items": a JSON list where each element has "query" (string) and "quantity" (integer). One element per product. Default quantity is 1.
- For add_all, return "quantity": an integer representing how many of EACH product to add (default 1 if not specified).
- For remove_from_cart: ALWAYS return "items": a JSON list where each element has "query" (string) and "quantity" (integer or null). One element per product. null quantity = remove all units. Single-product removes use a one-element items list.
- "quantity" values MUST be positive integers (>= 1). If the user says a negative number, return unsupported.
- For search: "query" is the search string.
- "yes" / "sure" / "ok" in reply to a bot suggestion to add SPECIFIC named products → add_to_cart for those exact products only, NOT add_all.
- "yes" / "ok" with no clear product context → chitchat.
- "add everything", "add all products", "add the whole inventory" → add_all.
- "add everything … N units / N of each" → add_all with "quantity": N.
- "add all [category]" / "add every [category]" / "add N of all [category]" → add_category.
  Supports multiple categories: "add N of all [cat1] and [cat2]".
  "categories" = list of category names (lowercase canonical: vegetables/fruits/dairy/snacks/beverages/detergents/household/personal care).
  "quantity" = N (default 1 if not specified). Same quantity applies to all listed categories.
  CRITICAL: if a category word is present alongside "all/every/each", prefer add_category over add_all.
  Return shape: {"intent": "add_category", "categories": ["<cat1>", ...], "quantity": <int>}
- "buy all" (cart already has items, user wants to checkout) → place_order.
- "buy everything / the whole inventory" (user wants to stock the cart first) → add_all.
- "place order for [product]" / "order just [product]" / "checkout only [product]" → unsupported. Partial checkout is not supported; the whole cart is always ordered together.
- "add back" / "put it back" / "add it again" → add_to_cart with items: [{"query": "", "quantity": 1}].
- "remove all" / "remove everything" / "empty my cart" / "clear cart" → clear_cart (removes ALL items).
- "remove N of each" / "reduce all by N" / "remove N from each" / "remove N of everything" → reduce_all with "quantity": N.
- "reduce_all" return shape: {"intent": "reduce_all", "query": "", "quantity": <int>}
- "remove all [product]" where a specific product is named → remove_from_cart for that product.
- Any message asking to ADD, RESTOCK, or MODIFY the store's inventory / catalogue → unsupported.
  Examples: "add spinach to inventory", "add mangoes to the store", "restock milk", "increase stock of apples".
  These are admin operations — the bot is a shopper assistant, not an inventory manager.

Return shape for add_to_cart:
{"intent": "add_to_cart", "items": [{"query": "<name>", "quantity": <int>}, ...], "query": "", "quantity": null}

Return shape for remove_from_cart:
{"intent": "remove_from_cart", "items": [{"query": "<name>", "quantity": <int or null>}, ...], "query": "", "quantity": null}

Return shape for add_all:
{"intent": "add_all", "query": "", "quantity": <int — per-product qty, default 1>}

Return shape for all other intents:
{"intent": "<intent>", "query": "<string>", "quantity": <integer or null>}

Examples:
"add 3 amul butter" → {"intent": "add_to_cart", "items": [{"query": "amul butter", "quantity": 3}], "query": "", "quantity": null}
"add 1 lays, 2 watermelon and 4 butter" → {"intent": "add_to_cart", "items": [{"query": "lays", "quantity": 1}, {"query": "watermelon", "quantity": 2}, {"query": "butter", "quantity": 4}], "query": "", "quantity": null}
"add 2 coke and 1 redbull" → {"intent": "add_to_cart", "items": [{"query": "coke", "quantity": 2}, {"query": "redbull", "quantity": 1}], "query": "", "quantity": null}
"add milk" → {"intent": "add_to_cart", "items": [{"query": "milk", "quantity": 1}], "query": "", "quantity": null}
"remove tomatoes" → {"intent": "remove_from_cart", "items": [{"query": "tomatoes", "quantity": null}], "query": "", "quantity": null}
"remove 2 units" → {"intent": "remove_from_cart", "items": [{"query": "", "quantity": 2}], "query": "", "quantity": null}
"remove 7 spinach and 7 apples" → {"intent": "remove_from_cart", "items": [{"query": "spinach", "quantity": 7}, {"query": "apples", "quantity": 7}], "query": "", "quantity": null}
"what's in my cart" → {"intent": "view_cart", "query": "", "quantity": null}
"buy all" → {"intent": "place_order", "query": "", "quantity": null}
"place order for harpic" → {"intent": "unsupported", "query": "place order for harpic", "quantity": null}
"order just the milk" → {"intent": "unsupported", "query": "order just the milk", "quantity": null}
"add everything to the cart" → {"intent": "add_all", "query": "", "quantity": 1}
"add everything available 5 units" → {"intent": "add_all", "query": "", "quantity": 5}
"add all products, 3 each" → {"intent": "add_all", "query": "", "quantity": 3}
"buy the whole inventory" → {"intent": "add_all", "query": "", "quantity": 1}
"buy all except onions" → {"intent": "unsupported", "query": "buy all except onions", "quantity": null}
"my budget is 500" → {"intent": "unsupported", "query": "my budget is 500", "quantity": null}
"add spinach to inventory" → {"intent": "unsupported", "query": "add spinach to inventory", "quantity": null}
"restock butter" → {"intent": "unsupported", "query": "restock butter", "quantity": null}
"remove all" → {"intent": "clear_cart", "query": "", "quantity": null}
"empty my cart" → {"intent": "clear_cart", "query": "", "quantity": null}
"remove all tomatoes" → {"intent": "remove_from_cart", "items": [{"query": "tomatoes", "quantity": null}], "query": "", "quantity": null}
"vegetables available" → {"intent": "search", "query": "vegetables", "quantity": null}
"do you have coke" → {"intent": "search", "query": "coke", "quantity": null}
"what products do you have" → {"intent": "list_all", "query": "", "quantity": null}
"add it back" → {"intent": "add_to_cart", "items": [{"query": "", "quantity": 1}], "query": "", "quantity": null}
"remove 2 of each" → {"intent": "reduce_all", "query": "", "quantity": 2}
"reduce all by 3" → {"intent": "reduce_all", "query": "", "quantity": 3}
"remove 1 from each item" → {"intent": "reduce_all", "query": "", "quantity": 1}
"add 5 items of all detergents" → {"intent": "add_category", "categories": ["detergents"], "quantity": 5}
"add all dairy products" → {"intent": "add_category", "categories": ["dairy"], "quantity": 1}
"add 2 of every vegetable" → {"intent": "add_category", "categories": ["vegetables"], "quantity": 2}
"add all snacks, 3 each" → {"intent": "add_category", "categories": ["snacks"], "quantity": 3}
"get me all beverages" → {"intent": "add_category", "categories": ["beverages"], "quantity": 1}
"add 2 of all fruits and detergents" → {"intent": "add_category", "categories": ["fruits", "detergents"], "quantity": 2}
"add all vegetables and dairy" → {"intent": "add_category", "categories": ["vegetables", "dairy"], "quantity": 1}
"""

RESPONSE_PROMPT = """You are Inventory tracker's shopping assistant. Write a short, direct response based only on the backend result below.
- Format prices as Rs X.XX
- Do NOT invent data not in the result.
- Be concise: 1–3 sentences unless showing a list.
- If the result says cart is empty, say exactly: Your cart is empty.
- Do NOT add suggestions, follow-up questions, or offers to help further.
"""

def _llm_call(system: str, user: str, max_tokens: int = 256) -> str:
    resp = _groq().chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


_ADMIN_OP_RE = re.compile(
    r"\b(add|put|restock|increase|update|edit|change)\b.{0,40}"
    r"\b(inventory|catalogue|catalog|store|stock)\b",
    re.IGNORECASE,
)

_PARTIAL_ORDER_RE = re.compile(
    r"\b(place\s+order|checkout)\b.{0,20}\bfor\b(?!\s+(tomorrow|today|tonight|now|later|delivery|me\b))"
    r"|\b(order|checkout)\s+(just|only)\b",
    re.IGNORECASE,
)


def _extract_intent(message: str, history: list[dict]) -> dict:
    if _ADMIN_OP_RE.search(message):
        return {"intent": "unsupported", "query": message, "quantity": None, "items": []}
    if _PARTIAL_ORDER_RE.search(message):
        return {"intent": "unsupported", "query": message, "quantity": None, "items": []}

    ctx = ""
    if history:
        recent = history[-4:]
        ctx = "Recent conversation:\n"
        ctx += "\n".join(f"{h['role']}: {h['content']}" for h in recent)
        ctx += "\n\n"

    raw = _llm_call(INTENT_PROMPT, f"{ctx}User message: {message}")
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        retry_prompt = (
            "Return ONLY a JSON object with keys intent, query, quantity. "
            "No markdown, no explanation.\n"
            f"Message: {message}"
        )
        raw2 = _llm_call(INTENT_PROMPT, retry_prompt)
        raw2 = re.sub(r"```json|```", "", raw2).strip()
        try:
            parsed = json.loads(raw2)
        except json.JSONDecodeError:
            parsed = _rule_based_fallback(message)

    if parsed.get("intent") == "add_to_cart":
        if not parsed.get("items"):
            q   = (parsed.get("query") or "").strip()
            qty = int(parsed.get("quantity") or 1)
            parsed["items"] = [{"query": q, "quantity": qty}]
        clean = []
        for item in parsed["items"]:
            clean.append({
                "query":    (item.get("query") or "").strip(),
                "quantity": max(1, int(item.get("quantity") or 1)),
            })
        parsed["items"] = clean

    if parsed.get("intent") == "remove_from_cart":
        if not parsed.get("items"):
            q   = (parsed.get("query") or "").strip()
            qty = parsed.get("quantity")
            parsed["items"] = [{"query": q, "quantity": qty}]
        clean = []
        for item in parsed["items"]:
            raw_q   = (item.get("query") or "").strip()
            raw_qty = item.get("quantity")
            clean.append({
                "query":    raw_q,
                "quantity": int(raw_qty) if raw_qty is not None else None,
            })
        parsed["items"] = clean

    if parsed.get("intent") == "add_category":
        if not parsed.get("categories"):
            q = (parsed.get("query") or "").strip()
            parsed["categories"] = [q] if q else []

    parsed.setdefault("query",      "")
    parsed.setdefault("quantity",   None)
    parsed.setdefault("items",      [])
    parsed.setdefault("categories", [])
    return parsed


def _rule_based_fallback(message: str) -> dict:
    m = message.strip().lower()

    if re.search(r"\b(restock|inventory|catalogue|catalog|store)\b", m) and re.search(r"\b(add|put|update|edit|change|increase|stock)\b", m):
        return {"intent": "unsupported", "query": message, "quantity": None}

    if re.search(r"\b(cart|basket)\b", m) and re.search(r"\b(what|show|view|see)\b", m):
        return {"intent": "view_cart", "query": "", "quantity": None}
    if re.search(r"\b(place|checkout|order|buy all)\b", m):
        return {"intent": "place_order", "query": "", "quantity": None}
    if re.search(r"\b(history|past order)\b", m):
        return {"intent": "order_history", "query": "", "quantity": None}

    _CATS    = ["vegetables","fruits","dairy","snacks","beverages","detergents","household","personal care"]
    _ADD_KW  = {"add", "get", "want", "give", "put", "buy"}
    _BULK_KW = {"all", "every", "each", "entire"}
    msg_tokens = set(m.split())
    has_add    = bool(msg_tokens & _ADD_KW)
    has_bulk   = bool(msg_tokens & _BULK_KW) or bool(re.search(r"\d+", m))
    found_cats = [c for c in _CATS if c in m]
    if has_add and has_bulk and found_cats:
        nums = re.findall(r"\d+", m)
        qty  = int(nums[0]) if nums else 1
        return {"intent": "add_category", "categories": found_cats, "quantity": qty}

    reduce_match = re.search(r"remove\s+(\d+).*\b(each|all|every|everything)\b|reduce.*\ball\b.*by\s+(\d+)|\b(each|every)\b.*remove\s+(\d+)", m)
    if reduce_match:
        nums = [g for g in reduce_match.groups() if g and g.isdigit()]
        qty = int(nums[0]) if nums else 1
        return {"intent": "reduce_all", "query": "", "quantity": qty}

    if re.search(r"\b(clear|empty|reset)\b.*\bcart\b", m):
        return {"intent": "clear_cart", "query": "", "quantity": None}
    if re.match(r"remove\s+all\s*$", m) or re.match(r"remove\s+everything\s*$", m):
        return {"intent": "clear_cart", "query": "", "quantity": None}

    add_match = re.match(r"add\s+(\d+)\s+(.+)", m)
    if add_match:
        qty   = int(add_match.group(1))
        query = add_match.group(2).strip()
        return {"intent": "add_to_cart", "items": [{"query": query, "quantity": qty}],
                "query": "", "quantity": None}
    if re.search(r"\badd\b", m):
        query = re.sub(r"^add\s+", "", m).strip()
        return {"intent": "add_to_cart", "items": [{"query": query, "quantity": 1}],
                "query": "", "quantity": None}

    return {"intent": "chitchat", "query": message, "quantity": None}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _depluralise(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def _apply_synonyms(query: str) -> str:
    norm = query.lower().strip()
    for alias, canonical in SYNONYMS.items():
        if alias in norm:
            norm = norm.replace(alias, canonical)
    return norm


def _query_tokens(query: str) -> set[str]:
    stopwords = {
        "add", "remove", "get", "buy", "want", "give", "me", "a",
        "an", "the", "some", "of", "to", "please", "i", "my", "put",
        "can", "could", "would", "do", "you", "have", "is", "are",
        "in", "into", "cart", "basket", "order",
    }
    expanded = _apply_synonyms(query)
    tokens = _normalize(expanded).split()
    stems: set[str] = set()
    for t in tokens:
        if t not in stopwords and len(t) > 1:
            stems.add(t)
            stems.add(_depluralise(t))
    return stems


def _product_tokens(product: dict) -> set[str]:
    blob = " ".join([
        product.get("name",        ""),
        product.get("category",    ""),
        product.get("description", ""),
    ])
    tokens = _normalize(blob).split()
    stems: set[str] = set()
    for t in tokens:
        if len(t) > 1:
            stems.add(t)
            stems.add(_depluralise(t))
    return stems


def _score_product(q_stems: set[str], product: dict) -> int:
    name_cat_blob = " ".join([product.get("name", ""), product.get("category", "")])
    name_cat_toks: set[str] = set()
    for t in _normalize(name_cat_blob).split():
        name_cat_toks.add(t)
        name_cat_toks.add(_depluralise(t))

    desc_toks: set[str] = set()
    for t in _normalize(product.get("description", "")).split():
        desc_toks.add(t)
        desc_toks.add(_depluralise(t))

    squished = _normalize(name_cat_blob).replace(" ", "")

    score = 0
    for qt in q_stems:
        if qt in name_cat_toks:
            score += 4
        elif qt in desc_toks:
            score += 3
        elif len(qt) >= 4:
            if any(len(pt) >= 4 and (pt.startswith(qt) or qt.startswith(pt))
                   for pt in name_cat_toks):
                score += 2
            elif qt in squished:
                score += 2
            elif any(len(pt) >= 4 and (pt.startswith(qt) or qt.startswith(pt))
                     for pt in desc_toks):
                score += 1
    return score


MIN_RESOLVE_SCORE = 2


def _resolve_product(
    query: str,
    sid: str | None = None,
    history: list[dict] | None = None,
    use_cart_fallback: bool = True,
) -> dict | None:
    if query.strip():
        q_stems = _query_tokens(query)
        if not q_stems:
            return None

        rag_hits = rag.retrieve(query, top_k=6)
        kw_hits  = inv.search_products(query)
        seen: set[str] = {p["product_id"] for p in rag_hits}
        candidates = [dict(p) for p in rag_hits]
        for p in kw_hits:
            if p["product_id"] not in seen:
                candidates.append(dict(p))
                seen.add(p["product_id"])

        best: dict | None = None
        best_score = 0

        for p in candidates:
            s = _score_product(q_stems, p)
            if s > best_score:
                best_score = s
                best = p

        if best_score < MIN_RESOLVE_SCORE:
            for p in inv.get_all_products():
                if p["product_id"] in seen:
                    continue
                s = _score_product(q_stems, p)
                if s > best_score:
                    best_score = s
                    best = p

        if best and best_score >= MIN_RESOLVE_SCORE:
            return inv.get_product(best["product_id"])

        return None

    if history:
        all_prods = inv.get_all_products()
        for turn in reversed(history[-6:]):
            content = _normalize(turn.get("content", ""))
            for p in all_prods:
                if _normalize(p["name"]) in content:
                    return inv.get_product(p["product_id"])

    if use_cart_fallback and sid:
        cart_items = crt.view_cart(sid)
        if cart_items:
            return inv.get_product(cart_items[-1]["product_id"])

    return None


CATEGORIES = {
    "vegetables", "fruits", "dairy", "snacks",
    "beverages", "detergents", "household", "personal care",
}


def _dispatch(intent: dict, sid: str, history: list[dict] | None = None, message: str = "") -> str:
    action  = intent.get("intent", "chitchat")
    query   = (intent.get("query") or "").strip()
    raw_qty = intent.get("quantity")

    if action == "list_all":
        products = inv.get_all_products()
        if not products:
            return "Inventory is empty."
        lines: list[str] = []
        cur_cat = ""
        for p in products:
            if p["category"] != cur_cat:
                cur_cat = p["category"]
                lines.append(f"\n{cur_cat}:")
            stock = "Out of Stock" if p["quantity"] == 0 else f"{p['quantity']} in stock"
            lines.append(f"  [{p['product_id']}] {p['name']} - Rs{p['price']} ({stock})")
        return "\n".join(lines)

    elif action == "search":
        _LIST_ALL_KW = {
            "all", "everything", "products", "items", "inventory",
            "available", "list", "show", "catalogue", "catalog",
        }
        if query and not (set(_normalize(query).split()) - _LIST_ALL_KW):
            return _dispatch({"intent": "list_all"}, sid, history, message)

        if not query:
            cart_items = crt.view_cart(sid)
            if not cart_items:
                return "What are you looking for?"
            products = [inv.get_product(i["product_id"]) for i in cart_items]
            products = [p for p in products if p]
        elif query.lower() in CATEGORIES:
            msg_norm   = _normalize(message)
            cart_scoped = "cart" in msg_norm or "my" in msg_norm
            if cart_scoped:
                cart_items = crt.view_cart(sid)
                in_cat = [i for i in cart_items if i["category"].lower() == query.lower()]
                if not in_cat:
                    return f"No {query} items in your cart."
                products = [inv.get_product(i["product_id"]) for i in in_cat]
                products = [p for p in products if p]
            else:
                products = inv.search_products(category=query)
        else:
            rag_hits = rag.retrieve(query, top_k=4)
            kw_hits  = inv.search_products(query)
            seen_ids: set[str] = {p["product_id"] for p in rag_hits}
            products = [dict(p) for p in rag_hits]
            for p in kw_hits:
                if p["product_id"] not in seen_ids:
                    products.append(dict(p))
            products = products[:5]

            if not products:
                found = _resolve_product(query, sid, history, use_cart_fallback=False)
                if found:
                    products = [found]
                else:
                    return (
                        f"Sorry, we don't carry '{query}'. "
                        "Type 'show all' to see what's available."
                    )

        fresh: list[dict] = []
        for p in products:
            fp = inv.get_product(p["product_id"])
            if fp:
                fresh.append(fp)

        if not fresh:
            return (
                f"Sorry, we don't carry '{query}'. "
                "Type 'show all' to see what's available."
            )

        cart_items = crt.view_cart(sid)
        cart_map   = {i["product_id"]: i["quantity"] for i in cart_items}
        adjusted: list[dict] = []
        for p in fresh:
            p2 = dict(p)
            p2["quantity"] = p["quantity"] - cart_map.get(p["product_id"], 0)
            adjusted.append(p2)

        return rag.format_context(adjusted)

    elif action == "add_to_cart":
        items = intent.get("items") or []
        if not items:
            items = [{"query": query, "quantity": raw_qty or 1}]

        added_lines:  list[str] = []
        failed_lines: list[str] = []

        for item in items:
            sq  = (item.get("query") or "").strip()
            qty = max(1, int(item.get("quantity") or 1))

            product = _resolve_product(sq, sid, history, use_cart_fallback=False)
            if not product:
                label = sq if sq else "(unknown product)"
                failed_lines.append(f"'{label}' not found")
                continue

            result = crt.add_to_cart(sid, product["product_id"], qty)
            if result["ok"]:
                fresh_p   = inv.get_product(product["product_id"])
                db_stock  = fresh_p["quantity"] if fresh_p else 0
                cart_now  = crt.view_cart(sid)
                in_cart   = next(
                    (i["quantity"] for i in cart_now
                     if i["product_id"] == product["product_id"]), 0
                )
                remaining = db_stock - in_cart
                added_lines.append(f"{product['name']} x{qty} ({remaining} left in stock)")
            else:
                failed_lines.append(f"{product['name']}: {result['reason']}")

        parts: list[str] = []
        if added_lines:
            parts.append("Added to cart: " + ", ".join(added_lines) + ".")
        if failed_lines:
            parts.append("Could not add: " + ", ".join(failed_lines) + ".")
        return " ".join(parts) if parts else "Nothing could be added."

    elif action == "add_all":
        per_qty  = max(1, int(raw_qty or 1))
        products = inv.get_all_products()
        if not products:
            return "Inventory is empty, nothing to add."
        added:   list[str] = []
        skipped: list[str] = []
        for p in products:
            fp = inv.get_product(p["product_id"])
            if not fp or fp["quantity"] == 0:
                skipped.append(p["name"])
                continue
            to_add = min(per_qty, fp["quantity"])
            result = crt.add_to_cart(sid, fp["product_id"], to_add)
            if result["ok"]:
                added.append(fp["name"])
            else:
                skipped.append(fp["name"])
        parts: list[str] = []
        if added:
            parts.append(
                f"Added {len(added)} product(s) to cart"
                f"{' (×' + str(per_qty) + ' each)' if per_qty > 1 else ''}"
                f": {', '.join(added)}."
            )
        if skipped:
            parts.append(f"Skipped (out of stock or already in cart): {', '.join(skipped)}.")
        return " ".join(parts) if parts else "Nothing could be added."

    elif action == "add_category":
        per_qty  = max(1, int(raw_qty or 1))
        raw_cats = intent.get("categories") or []
        if not raw_cats and query.strip():
            raw_cats = [query.strip().lower()]

        def match_cat(raw: str) -> str | None:
            c = raw.strip().lower()
            if c in CATEGORIES:
                return c
            for cat in CATEGORIES:
                if c in cat or cat in c:
                    return cat
            return None

        matched_cats: list[str] = []
        unrecognised: list[str] = []
        for rc in raw_cats:
            mc = match_cat(rc)
            if mc:
                if mc not in matched_cats:
                    matched_cats.append(mc)
            else:
                unrecognised.append(rc)

        if not matched_cats:
            return (
                f"I don't recognise those as categories. "
                f"Available: {', '.join(sorted(CATEGORIES))}."
            )

        added:   list[str] = []
        skipped: list[str] = []
        for cat in matched_cats:
            for p in inv.search_products(category=cat):
                fp = inv.get_product(p["product_id"])
                if not fp or fp["quantity"] == 0:
                    skipped.append(p["name"])
                    continue
                to_add = min(per_qty, fp["quantity"])
                result = crt.add_to_cart(sid, fp["product_id"], to_add)
                if result["ok"]:
                    added.append(fp["name"])
                else:
                    skipped.append(fp["name"])

        cat_labels = " & ".join(c.title() for c in matched_cats)
        qty_suffix = f" (\u00d7{per_qty} each)" if per_qty > 1 else ""
        parts: list[str] = []
        if added:
            parts.append(
                f"Added {len(added)} {cat_labels} product(s) to cart{qty_suffix}"
                f": {', '.join(added)}."
            )
        if skipped:
            parts.append(f"Skipped (out of stock or already maxed): {', '.join(skipped)}.")
        if unrecognised:
            parts.append(f"Didn't recognise: {', '.join(unrecognised)}.")
        return " ".join(parts) if parts else "Nothing could be added."

    elif action == "remove_from_cart":
        items = intent.get("items") or []
        removed_lines:  list[str] = []
        updated_lines:  list[str] = []
        not_found_lines: list[str] = []

        for item in items:
            iq  = (item.get("query") or "").strip()
            iqt = item.get("quantity")

            if iq.lower() in CATEGORIES:
                cat        = iq.lower()
                cart_items = crt.view_cart(sid)
                targets    = [i for i in cart_items if i["category"].lower() == cat]
                if not targets:
                    not_found_lines.append(f"no {cat} items in cart")
                    continue
                for ci in targets:
                    if iqt is None:
                        r = crt.remove_from_cart(sid, ci["product_id"])
                        if r["ok"]:
                            removed_lines.append(ci["name"])
                    else:
                        r = crt.reduce_cart_quantity(sid, ci["product_id"], iqt)
                        if r["ok"]:
                            if r["new_qty"] == 0:
                                removed_lines.append(ci["name"])
                            else:
                                updated_lines.append(f"{ci['name']} → {r['new_qty']}")
                continue

            product = _resolve_product(iq, sid, history, use_cart_fallback=len(items)==1)
            if not product:
                label = iq if iq else "(unknown product)"
                not_found_lines.append(f"'{label}' not found")
                continue
            pid        = product["product_id"]
            cart_items = crt.view_cart(sid)
            in_cart    = next((i for i in cart_items if i["product_id"] == pid), None)
            if not in_cart:
                not_found_lines.append(f"{product['name']} not in cart")
                continue
            if iqt is None:
                r = crt.remove_from_cart(sid, pid)
                if r["ok"]:
                    removed_lines.append(product["name"])
            else:
                if iqt <= 0:
                    not_found_lines.append(f"invalid quantity for {product['name']}")
                    continue
                r = crt.reduce_cart_quantity(sid, pid, iqt)
                if not r["ok"]:
                    not_found_lines.append(f"{product['name']}: {r['reason']}")
                elif r["new_qty"] == 0:
                    removed_lines.append(product["name"])
                else:
                    updated_lines.append(f"{product['name']} → {r['new_qty']}")

        parts: list[str] = []
        if removed_lines:
            parts.append("Removed: " + ", ".join(removed_lines) + ".")
        if updated_lines:
            parts.append("Updated: " + ", ".join(updated_lines) + ".")
        if not_found_lines:
            parts.append("Could not process: " + ", ".join(not_found_lines) + ".")
        return " ".join(parts) if parts else "Nothing changed."

    elif action == "reduce_all":
        reduce_by  = max(1, int(raw_qty or 1))
        cart_items = crt.view_cart(sid)
        if not cart_items:
            return "Your cart is empty."
        msg_norm   = _normalize(message)
        scoped_cat = next((c for c in CATEGORIES if c in msg_norm), None)
        if scoped_cat:
            cart_items = [i for i in cart_items if i["category"].lower() == scoped_cat]
            if not cart_items:
                return f"No {scoped_cat} items in your cart."
        updated: list[str] = []
        removed: list[str] = []
        for item in cart_items:
            result = crt.reduce_cart_quantity(sid, item["product_id"], reduce_by)
            if result["ok"]:
                if result["new_qty"] == 0:
                    removed.append(item["name"])
                else:
                    updated.append(f"{item['name']} → {result['new_qty']}")
        parts: list[str] = []
        if updated:
            parts.append("Updated: " + ", ".join(updated) + ".")
        if removed:
            parts.append("Removed entirely: " + ", ".join(removed) + ".")
        return " ".join(parts) if parts else "Nothing changed."

    elif action == "clear_cart":
        crt.clear_cart(sid)
        return "Cart cleared."

    elif action == "view_cart":
        items = crt.view_cart(sid)
        if not items:
            return "Cart is empty."
        lines = ["Cart contents:"]
        grand_total = 0.0
        for item in items:
            lines.append(
                f"  [{item['product_id']}] {item['name']} "
                f"x{item['quantity']} @ Rs{item['price']} = Rs{item['subtotal']:.2f}"
            )
            grand_total += item["subtotal"]
        lines.append(f"Grand Total: Rs{grand_total:.2f}")
        return "\n".join(lines)

    elif action == "place_order":
        result = ord_.place_order(sid)
        if result["ok"]:
            lines = ["Order placed!"]
            for o in result["orders"]:
                lines.append(f"  {o['name']} x{o['quantity']} = Rs{o['total']:.2f}")
            lines.append(f"Total: Rs{result['total']:.2f}")
            return "\n".join(lines)
        return f"Order failed: {result['reason']}"

    elif action == "order_history":
        history_ = ord_.get_order_history(sid)
        if not history_:
            return "No orders placed yet."
        lines = ["Order history:"]
        for o in history_:
            lines.append(
                f"  #{o['order_id']}: {o['name']} x{o['quantity']} "
                f"= Rs{o['total']:.2f} on {o['placed_at'][:10]}"
            )
        return "\n".join(lines)

    elif action == "unsupported":
        q = (intent.get("query") or "").lower()
        if _PARTIAL_ORDER_RE.search(q) or _PARTIAL_ORDER_RE.search(message):
            return (
                "Partial checkout isn't supported — place order places everything "
                "in your cart at once. Remove items you don't want first, then place order."
            )
        return (
            "I can't help with that. I support searching products, "
            "managing your cart, and placing orders — no budgets, coupons, "
            "or partial checkouts."
        )

    return "CHITCHAT"


_DIRECT_INTENTS = {
    "add_to_cart", "remove_from_cart", "clear_cart",
    "view_cart", "place_order", "order_history", "unsupported",
    "search", "list_all", "add_all", "add_category", "reduce_all",
}

_YES_TOKENS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead",
               "do it", "add them", "add those", "yes please", "sure please"}


def _is_affirmative(message: str) -> bool:
    norm = _normalize(message)
    if norm in _YES_TOKENS:
        return True
    for yw in _YES_TOKENS:
        if norm.startswith(yw) and (len(norm) == len(yw) or norm[len(yw)] == " "):
            return True
    return False


def _pending_specific_products(history: list[dict]) -> list[dict] | None:
    if not history:
        return None
    last_bot = next(
        (h["content"] for h in reversed(history) if h["role"] == "assistant"), ""
    )
    if not last_bot:
        return None

    lower = _normalize(last_bot)
    if any(kw in lower for kw in ("add all", "whole inventory", "all products", "entire inventory")):
        return None

    all_prods = inv.get_all_products()
    found: list[dict] = []
    seen_ids: set[str] = set()
    for p in all_prods:
        if _normalize(p["name"]) in lower and p["product_id"] not in seen_ids:
            found.append(p)
            seen_ids.add(p["product_id"])
    return found if found else None


def chat(sid: str, message: str, history: list[dict]) -> str:
    intent = _extract_intent(message, history)
    action = intent.get("intent", "chitchat")

    if action == "add_all" and _is_affirmative(message):
        specific = _pending_specific_products(history)
        if specific:
            intent = {
                "intent":   "add_to_cart",
                "items":    [{"query": p["name"], "quantity": 1} for p in specific],
                "query":    "",
                "quantity": None,
            }
            action = "add_to_cart"

    backend_result = _dispatch(intent, sid, history, message)

    if backend_result == "CHITCHAT":
        ctx = ""
        if history:
            recent = history[-4:]
            ctx = "\nRecent conversation:\n" + "\n".join(
                f"{h['role']}: {h['content']}" for h in recent
            )
        return _llm_call(
            "You are Inventory tracker's shopping assistant. "
            "Keep answers brief and relevant to the conversation. "
            "Never claim to have modified the cart or inventory — "
            "only the backend actions do that.",
            f"{ctx}\n\nUser: {message}",
            max_tokens=200,
        )

    if action in _DIRECT_INTENTS:
        return backend_result

    return _llm_call(
        RESPONSE_PROMPT,
        f"Backend result:\n{backend_result}",
        max_tokens=512,
    )