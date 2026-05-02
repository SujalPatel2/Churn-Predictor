import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np
import requests
from data_loader import load_data, get_summary
from model_trainer import preprocess, load_model
from sklearn.preprocessing import LabelEncoder

def get_ai_insights(summary, top_features):
    prompt = f"""
You are an expert HR and business analyst. Analyze this customer churn data and give 4 short bullet point insights and 1 recommendation.

Data:
- Total Customers: {summary['total']}
- Churned: {summary['churned']} ({summary['churn_rate']}%)
- Retained: {summary['stayed']}
- Top factors causing churn: {', '.join(top_features)}

Give response in this exact format:
INSIGHT 1: ...
INSIGHT 2: ...
INSIGHT 3: ...
INSIGHT 4: ...
RECOMMENDATION: ...
"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI insights unavailable: {str(e)}"


def show_dashboard():
    st.title("📊 Customer Churn Analytics")
    st.markdown(f"Welcome, **{st.session_state['username']}** 👋")

    df_raw = load_data()
    summary = get_summary(df_raw)
    model, feature_cols = load_model()
    df_proc = preprocess(load_data())

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 EDA Dashboard",
        "🤖 Churn Predictor",
        "🚨 At-Risk Customers",
        "✨ AI Insights"
    ])

    # ── TAB 1: OVERVIEW ──────────────────────────────────────
    with tab1:
        st.subheader("📌 Key Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", summary["total"])
        c2.metric("Churned", summary["churned"])
        c3.metric("Retained", summary["stayed"])
        c4.metric("Churn Rate", f"{summary['churn_rate']}%")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                values=[summary["stayed"], summary["churned"]],
                names=["Retained", "Churned"],
                title="Churn Distribution",
                color_discrete_sequence=["#00CC96", "#EF553B"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            contract_churn = df_raw.groupby("Contract")["Churn"].mean().reset_index()
            contract_churn["Churn"] = (contract_churn["Churn"] * 100).round(2)
            fig2 = px.bar(
                contract_churn, x="Contract", y="Churn",
                title="Churn Rate by Contract Type (%)",
                color="Churn", color_continuous_scale="Reds"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── FEATURE IMPORTANCE CHART ──
        st.subheader("🔍 What Causes Churn the Most?")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=True).tail(10)

        fig_imp = px.bar(
            feat_df, x="Importance", y="Feature",
            orientation="h",
            title="Top 10 Churn Factors (Feature Importance)",
            color="Importance",
            color_continuous_scale=[[0, "#2d2d2d"], [1, "#FFD700"]]
        )
        fig_imp.update_layout(
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font_color="#ffffff"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── TAB 2: EDA ────────────────────────────────────────────
    with tab2:
        st.subheader("📈 Exploratory Analysis")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df_raw, x="tenure",
                color=df_raw["Churn"].map({1: "Churned", 0: "Stayed"}),
                title="Tenure Distribution by Churn",
                barmode="overlay",
                color_discrete_map={"Churned": "#EF553B", "Stayed": "#00CC96"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                df_raw, x=df_raw["Churn"].map({1: "Churned", 0: "Stayed"}),
                y="MonthlyCharges",
                title="Monthly Charges vs Churn",
                color=df_raw["Churn"].map({1: "Churned", 0: "Stayed"}),
                color_discrete_map={"Churned": "#EF553B", "Stayed": "#00CC96"}
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(
                df_raw, x="InternetService",
                color=df_raw["Churn"].map({1: "Churned", 0: "Stayed"}),
                title="Internet Service vs Churn",
                barmode="group",
                color_discrete_map={"Churned": "#EF553B", "Stayed": "#00CC96"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.histogram(
                df_raw, x="PaymentMethod",
                color=df_raw["Churn"].map({1: "Churned", 0: "Stayed"}),
                title="Payment Method vs Churn",
                barmode="group",
                color_discrete_map={"Churned": "#EF553B", "Stayed": "#00CC96"}
            )
            fig.update_xaxes(tickangle=15)
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: PREDICTOR ─────────────────────────────────────
    with tab3:
        st.subheader("🤖 Predict Churn for a Customer")
        st.info("Fill in customer details and click Predict!")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

        if st.button("🔮 Predict Churn", use_container_width=True):
            input_dict = {
                "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone_service,
                "MultipleLines": multiple_lines, "InternetService": internet_service,
                "OnlineSecurity": online_security, "OnlineBackup": online_backup,
                "DeviceProtection": device_protection, "TechSupport": tech_support,
                "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }

            input_df = pd.DataFrame([input_dict])
            le = LabelEncoder()
            for col in input_df.select_dtypes(include="object").columns:
                input_df[col] = le.fit_transform(input_df[col])

            input_df = input_df[feature_cols]
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.divider()

            # ── GAUGE CHART ──
            col_g1, col_g2 = st.columns([1, 2])
            with col_g1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(probability * 100, 1),
                    title={"text": "Churn Risk %", "font": {"color": "#FFD700"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#FFD700"},
                        "bar": {"color": "#FFD700"},
                        "bgcolor": "#2d2d2d",
                        "steps": [
                            {"range": [0, 40], "color": "#1a3a1a"},
                            {"range": [40, 70], "color": "#3a2a00"},
                            {"range": [70, 100], "color": "#3a1a1a"}
                        ],
                        "threshold": {
                            "line": {"color": "#EF553B", "width": 4},
                            "thickness": 0.75,
                            "value": 70
                        }
                    },
                    number={"suffix": "%", "font": {"color": "#FFD700"}}
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="#1a1a1a",
                    font_color="#ffffff",
                    height=250
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_g2:
                if prediction == 1:
                    st.error(f"⚠️ This customer is LIKELY TO CHURN! (Risk: {round(probability*100, 1)}%)")
                else:
                    st.success(f"✅ This customer is likely to STAY! (Churn Risk: {round(probability*100, 1)}%)")
                st.progress(float(probability))

    # ── TAB 4: AT-RISK CUSTOMERS ─────────────────────────────
    with tab4:
        st.subheader("🚨 High-Risk Customers")

        X = df_proc[feature_cols]
        probs = model.predict_proba(X)[:, 1]
        df_raw2 = load_data()
        df_raw2["Churn Risk %"] = (probs * 100).round(1)

        threshold = st.slider("Show customers with risk above (%)", 50, 95, 70)
        high_risk = df_raw2[df_raw2["Churn Risk %"] > threshold].sort_values(
            "Churn Risk %", ascending=False
        )

        st.metric("High Risk Customers Found", len(high_risk))

        # ── DOWNLOAD CSV BUTTON ──
        csv = high_risk[["tenure", "Contract", "MonthlyCharges",
                          "InternetService", "PaymentMethod", "Churn Risk %"]].to_csv(index=False)
        st.download_button(
            label="⬇️ Download High-Risk Report (CSV)",
            data=csv,
            file_name="high_risk_customers.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.dataframe(
            high_risk[["tenure", "Contract", "MonthlyCharges",
                        "InternetService", "PaymentMethod", "Churn Risk %"]],
            use_container_width=True
        )

        fig = px.histogram(
            df_raw2, x="Churn Risk %",
            title="Distribution of Churn Risk Across All Customers",
            color_discrete_sequence=["#FFD700"]
        )
        fig.update_layout(
            plot_bgcolor="#1a1a1a",
            paper_bgcolor="#1a1a1a",
            font_color="#ffffff"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: AI INSIGHTS ───────────────────────────────────
    with tab5:
        st.subheader("✨ AI-Powered Churn Insights")
        st.info("Click the button below to generate smart insights using AI!")

        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        top_features = feat_df["Feature"].head(5).tolist()

        if st.button("🤖 Generate AI Insights", use_container_width=True):
            with st.spinner("AI is analyzing your churn data..."):
                insights = get_ai_insights(summary, top_features)

            st.divider()
            lines = insights.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("RECOMMENDATION:"):
                    st.success(f"💡 {line}")
                elif line.startswith("INSIGHT"):
                    st.info(f"📌 {line}")
                else:
                    st.write(line)