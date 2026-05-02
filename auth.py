import json
import hashlib
import os
import streamlit as st

USERS_FILE = "users.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def login_page():
    st.title("🔐 Customer Churn Predictor")
    st.subheader("Login to continue")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            users = load_users()
            if username in users and users[username] == hash_password(password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid username or password!")

    with tab2:
        new_user = st.text_input("Choose Username", key="signup_user")
        new_pass = st.text_input("Choose Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            users = load_users()
            if new_user in users:
                st.error("Username already exists!")
            else:
                users[new_user] = hash_password(new_pass)
                save_users(users)
                st.success("Account created! Please login.")