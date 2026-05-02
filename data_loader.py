import pandas as pd

def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # TotalCharges has some empty strings — fix them
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows where TotalCharges is null (only ~11 rows)
    df.dropna(inplace=True)

    # Convert Churn column: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID — not useful for analysis
    df.drop(columns=["customerID"], inplace=True)

    return df


def get_summary(df):
    total = len(df)
    churned = df["Churn"].sum()
    stayed = total - churned
    churn_rate = round((churned / total) * 100, 2)

    return {
        "total": total,
        "churned": int(churned),
        "stayed": int(stayed),
        "churn_rate": churn_rate
    }