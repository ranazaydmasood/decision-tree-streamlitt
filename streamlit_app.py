import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Starbucks Drink Decision Tree", page_icon="☕", layout="wide")
st.title("☕ Starbucks Drink Category Predictor")
st.write("Explore Starbucks drink nutrition data and predict a beverage category using a decision tree.")

# Load the dataset from the ZIP file shipped with the app.
DATA_ZIP = Path(__file__).parent / "archive.zip"
CSV_NAME = "starbucks_drinkMenu_expanded.csv"

@st.cache_data
def load_data():
    if not DATA_ZIP.exists():
        raise FileNotFoundError("archive.zip is missing. Add it beside app.py in your GitHub repository.")
    with zipfile.ZipFile(DATA_ZIP, "r") as archive:
        with archive.open(CSV_NAME) as file:
            data = pd.read_csv(file)
    data.columns = data.columns.str.strip()
    for col in ["Total Fat (g)", "Caffeine (mg)"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    pct_cols = ["Vitamin A (% DV)", "Vitamin C (% DV)", "Calcium (% DV)", "Iron (% DV)"]
    for col in pct_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace("%", "", regex=False), errors="coerce") / 100
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].median())
    return data

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

features = [
    "Calories", "Total Fat (g)", "Trans Fat (g)", "Saturated Fat (g)",
    "Sodium (mg)", "Total Carbohydrates (g)", "Cholesterol (mg)",
    "Dietary Fibre (g)", "Sugars (g)", "Protein (g)", "Caffeine (mg)"
]
missing = [c for c in features + ["Beverage_category"] if c not in df.columns]
if missing:
    st.error("Dataset is missing required columns: " + ", ".join(missing))
    st.stop()

X = df[features].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
y = df["Beverage_category"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

tab1, tab2, tab3 = st.tabs(["Overview", "Explore data", "Predict category"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Drinks in dataset", len(df))
    c2.metric("Beverage categories", df["Beverage_category"].nunique())
    c3.metric("Model accuracy", f"{accuracy:.1%}")
    st.subheader("Sample data")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Model evaluation")
    st.text(classification_report(y_test, predictions, zero_division=0))

with tab2:
    st.subheader("Calories distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Calories"], kde=True, ax=ax)
    ax.set_title("Calories in Starbucks drinks")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sugar vs calories")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Sugars (g)", y="Calories", hue="Beverage_category", ax=ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True)

with tab3:
    st.write("Enter nutrition values to estimate the drink's beverage category.")
    defaults = {
        "Calories": 150, "Total Fat (g)": 5.0, "Trans Fat (g)": 0.0,
        "Saturated Fat (g)": 3.0, "Sodium (mg)": 20.0,
        "Total Carbohydrates (g)": 20.0, "Cholesterol (mg)": 15.0,
        "Dietary Fibre (g)": 0.0, "Sugars (g)": 18.0,
        "Protein (g)": 5.0, "Caffeine (mg)": 100.0
    }
    inputs = {}
    with st.form("prediction_form"):
        cols = st.columns(2)
        for i, feature in enumerate(features):
            inputs[feature] = cols[i % 2].number_input(
                feature, min_value=0.0, value=float(defaults[feature]), step=1.0
            )
        submitted = st.form_submit_button("Predict category")
    if submitted:
        input_df = pd.DataFrame([inputs], columns=features)
        result = model.predict(input_df)[0]
        st.success(f"Predicted beverage category: {result}")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.write("Prediction confidence by category")
            st.dataframe(
                pd.DataFrame({"Category": model.classes_, "Probability": probabilities})
                .sort_values("Probability", ascending=False),
                use_container_width=True
            )
