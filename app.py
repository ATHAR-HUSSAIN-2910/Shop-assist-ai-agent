import logging
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from src.graph import graph


# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("shopassist.ui")


# ---------- Page config ----------
# initial_sidebar_state="expanded" => sidebar is open on desktop.
# On phones Streamlit renders it as a drawer you open with the ☰ toggle.
st.set_page_config(
    page_title="ShopAssist AI",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------- Custom CSS (professional look) ----------
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef2f8 100%);
    }

    /* ---- Header ---- */
    .sa-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 22px;
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        border-radius: 16px;
        color: white;
        box-shadow: 0 6px 24px rgba(79, 70, 229, 0.25);
        margin-bottom: 18px;
    }
    .sa-header h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .sa-header p {
        margin: 2px 0 0 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .sa-logo {
        font-size: 2rem;
        line-height: 1;
    }

    /* ---- Chat messages ---- */
    div[data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e5e9f2;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 10px;
    }

    /* ---- Quick action chips ---- */
    .sa-chips-title {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 6px 0 8px 0;
        font-weight: 600;
    }

    div.stButton > button {
        border-radius: 999px !important;
        border: 1px solid #dbe2ee !important;
        background: #ffffff !important;
        color: #1e293b !important;
        font-size: 0.83rem !important;
        padding: 6px 14px !important;
        transition: all 0.15s ease !important;
        font-weight: 500 !important;
    }
    div.stButton > button:hover {
        border-color: #6366f1 !important;
        color: #4f46e5 !important;
        background: #f5f3ff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e9f2;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .sa-side-card {
        background: #f8fafc;
        border: 1px solid #e5e9f2;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: #334155;
    }
    .sa-side-card strong {
        color: #0f172a;
        display: block;
        margin-bottom: 4px;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ---- Chat input ---- */
    div[data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 1px solid #dbe2ee !important;
    }

    /* ============================================================
       Streamlit chrome — hide the clutter, but KEEP the sidebar
       toggle alive (this is what makes the sidebar reachable on
       narrow laptop windows and on phones).
       ============================================================ */

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Hide the toolbar buttons (Deploy, ⋮ menu, "Running" status)
       without hiding the whole header. */
    [data-testid="stToolbar"] { visibility: hidden; height: 0; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stStatusWidget"] { visibility: hidden; }

    /* Header stays present (transparent) so the toggle inside it
       remains clickable. */
    [data-testid="stHeader"] {
        background: transparent;
        box-shadow: none;
    }

    /* Force the sidebar collapse / expand control to be visible
       and stacked above everything. Covers several Streamlit
       versions because the test-id has changed over time. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[kind="headerNoPadding"] {
        visibility: visible !important;
        display: flex !important;
        z-index: 1000000 !important;
    }

    /* Give the toggle a color that reads on the light background */
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg,
    button[kind="headerNoPadding"] svg {
        fill: #4f46e5 !important;
        color: #4f46e5 !important;
    }

    /* ---- Responsive ---- */
    @media (max-width: 768px) {
        .sa-header { padding: 14px 16px; }
        .sa-header h1 { font-size: 1.2rem; }
        .sa-header p { font-size: 0.75rem; }

        /* Sidebar becomes a comfortable drawer on phones */
        section[data-testid="stSidebar"] {
            min-width: 260px !important;
            width: 82vw !important;
            max-width: 320px !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Session state ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🛍️ ShopAssist AI")
    st.caption("Your AI customer support assistant")

    st.markdown(
        """
        <div class="sa-side-card">
            <strong>What I can do</strong>
            • Track your orders<br>
            • Cancel eligible orders<br>
            • Answer return, refund &amp; shipping questions<br>
            • Connect you to a human agent
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sa-side-card">
            <strong>Demo note</strong>
            Orders 2001–2300 are shared demo data.
            There is no authentication.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🔄 Start new conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


# ---------- Header ----------
st.markdown(
    """
    <div class="sa-header">
        <div class="sa-logo">🛍️</div>
        <div>
            <h1>ShopAssist AI</h1>
            <p>AI-powered customer support · Available 24/7</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------- Render existing chat ----------
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🛍️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ---------- Quick action chips (only on empty chat) ----------
if not st.session_state.messages:
    st.markdown(
        '<div class="sa-chips-title">Try asking</div>',
        unsafe_allow_html=True
    )

    chips = [
        "Where is my order 2002?",
        "What is your refund policy?",
        "I want to cancel order 2015",
        "I want to speak to a human agent",
    ]

    cols = st.columns(2)
    for i, chip in enumerate(chips):
        with cols[i % 2]:
            if st.button(chip, key=f"chip_{i}", use_container_width=True):
                # Only stage the input — do NOT append to messages here.
                # The input handler below is the single source of truth.
                st.session_state["_pending_input"] = chip
                st.rerun()


# ---------- Determine input (typed or from chip) ----------
user_input = st.chat_input("How can I help you today?")

if not user_input and st.session_state.get("_pending_input"):
    user_input = st.session_state.pop("_pending_input")


# ---------- Handle input ----------
if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    with st.chat_message("assistant", avatar="🛍️"):
        with st.spinner("Thinking..."):
            try:
                result = graph.invoke(initial_state, config=config)
                response = result.get(
                    "response",
                    "Sorry, I could not process your request."
                )

            except Exception as e:
                logger.error(
                    f"Graph invocation failed: {e}",
                    exc_info=True
                )
                response = (
                    "Sorry, I'm temporarily unable to process your request. "
                    "Please try again in a moment."
                )

            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )