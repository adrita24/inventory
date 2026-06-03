import os
import uuid
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from db import init_db
from rag import build_index
from cart import ensure_session, view_cart, clear_cart
from inventory import list_categories, search_products
import agent as ag

st.set_page_config(
    page_title="Inventory tracker",
    page_icon="🛒",
    layout="wide",
)

@st.cache_resource
def startup():
    init_db()
    build_index()

startup()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    ensure_session(st.session_state.session_id)

if "messages" not in st.session_state:
    st.session_state.messages = []

SESSION = st.session_state.session_id

with st.sidebar:
    st.title("Inventory tracker")
    st.caption(f"Session: `{SESSION[:8]}...`")

    st.divider()
    st.subheader("Cart")
    cart_items = view_cart(SESSION)
    if cart_items:
        running_total = 0.0
        for item in cart_items:
            st.write(f"**{item['name']}** x{item['quantity']} — ₹{item['subtotal']:.2f}")
            running_total += item["subtotal"]
        st.write(f"**Total: ₹{running_total:.2f}**")
        if st.button("Clear Cart"):
            clear_cart(SESSION)
            st.rerun()
    else:
        st.write("_Cart is empty_")

    st.divider()
    st.subheader("Browse by Category")
    cats = list_categories()
    picked_cat = st.selectbox("Category", ["All"] + cats)
    if picked_cat != "All":
        cat_products = search_products(category=picked_cat)
        for p in cat_products:
            dot = "🔴" if p["quantity"]==0 else ("🟡" if p["quantity"]<5 else "🟢")
            st.write(f"{dot} **{p['name']}** — ₹{p['price']} ({p['quantity']} left)")

    st.divider()
    if st.button("Reset Session"):
        clear_cart(SESSION)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        ensure_session(st.session_state.session_id)
        st.rerun()

st.title("Inventory tracker Shopping Assistant")
st.caption("Adrita Guha")

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not set. Add it in Streamlit Cloud secrets or a local .env file.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.write("**Quick actions:**")
cols = st.columns(4)
quick_actions = [
    ("Show all products", "What products do you have?"),
    ("View my cart",      "What's in my cart?"),
    ("Order history",     "Show my order history"),
    ("Place order",       "Buy everything in my cart"),
]
for col, (label, prompt) in zip(cols, quick_actions):
    if col.button(label):
        st.session_state._quick_prompt = prompt
        st.rerun()

if hasattr(st.session_state, "_quick_prompt") and st.session_state._quick_prompt:
    user_input = st.session_state._quick_prompt
    st.session_state._quick_prompt = None
else:
    user_input = st.chat_input("Ask me anything... e.g. 'Do you have detergent? Add 2 to cart'")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ag.chat(SESSION, user_input, st.session_state.messages[:-1])
            except Exception as e:
                response = f"Error: {e}"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()