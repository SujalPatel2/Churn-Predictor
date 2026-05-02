import streamlit as st

st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="📊",
    layout="wide"
)

# ── Premium Dark Grey + Gold CSS ──
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #1a1a1a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 2px solid #FFD700;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #2d2d2d;
    border: 1px solid #FFD700;
    border-radius: 10px;
    padding: 15px;
}

/* Metric label */
[data-testid="metric-container"] label {
    color: #FFD700 !important;
    font-weight: bold;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #2d2d2d;
    border-radius: 10px;
    padding: 5px;
}

.stTabs [data-baseweb="tab"] {
    color: #ffffff;
    border-radius: 8px;
}

.stTabs [aria-selected="true"] {
    background-color: #FFD700 !important;
    color: #1a1a1a !important;
    font-weight: bold;
}

/* Buttons */
.stButton > button {
    background-color: #FFD700;
    color: #1a1a1a;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    width: 100%;
}

.stButton > button:hover {
    background-color: #e6c200;
    color: #1a1a1a;
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background-color: #2d2d2d;
    color: #ffffff;
    border: 1px solid #FFD700;
    border-radius: 8px;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #FFD700;
    border-radius: 10px;
}

/* Title styling */
h1, h2, h3 {
    color: #FFD700 !important;
}

/* Divider */
hr {
    border-color: #FFD700;
}

/* Success/Error boxes */
.stSuccess {
    background-color: #1a3a1a;
    border-left: 4px solid #00CC96;
}

.stError {
    background-color: #3a1a1a;
    border-left: 4px solid #EF553B;
}

/* Login card */
.login-card {
    background-color: #2d2d2d;
    border: 1px solid #FFD700;
    border-radius: 15px;
    padding: 30px;
    max-width: 400px;
    margin: auto;
}

/* Gold badge */
.gold-badge {
    background-color: #FFD700;
    color: #1a1a1a;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

from auth import login_page
from dashboard import show_dashboard

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    # Sidebar branding
    with st.sidebar:
        st.markdown("## 📊 Churn Predictor")
        st.markdown('<span class="gold-badge">PRO</span>', unsafe_allow_html=True)
        st.markdown(f"👤 **{st.session_state['username']}**")
        st.divider()
        if st.button("🚪 Logout"):
            st.session_state["logged_in"] = False
            st.rerun()
    show_dashboard()
else:
    login_page()