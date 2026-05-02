import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from data_loader import load_data

def preprocess(df):
    df = df.copy()

    # Find all text (object) columns and encode them to numbers
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])

    return df


def train_model():
    df = load_data()
    df = preprocess(df)

    # X = all columns except Churn, y = Churn column
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Check accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {round(acc * 100, 2)}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model to file
    with open("churn_model.pkl", "wb") as f:
        pickle.dump((model, X.columns.tolist()), f)

    print("✅ Model saved as churn_model.pkl")

    return model, X.columns.tolist()


def load_model():
    with open("churn_model.pkl", "rb") as f:
        model, feature_cols = pickle.load(f)
    return model, feature_cols


if __name__ == "__main__":
    train_model()