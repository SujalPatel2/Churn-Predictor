import streamlit as st

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

from auth import login_page
from dashboard import show_dashboard

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    if st.sidebar.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.rerun()
    show_dashboard()
else:
    login_page()