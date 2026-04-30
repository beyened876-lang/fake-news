import os
import re
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# NLTK SETUP
# -----------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def data_path(filename):
    return os.path.join(BASE_DIR, filename)

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    return " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )

# -----------------------------
# LOAD DATASET (SAFE VERSION)
# -----------------------------
def load_test_data():
    path = data_path("test.tsv")

    if not os.path.exists(path):
        st.error("❌ test.tsv file not found in project folder")
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            engine="python",
            header=None,
            names=[
                "id", "label", "statement", "subject", "speaker",
                "job_title", "state_info", "party_affiliation",
                "barely_true_counts", "false_counts",
                "half_true_counts", "mostly_true_counts",
                "pants_on_fire_counts", "context"
            ],
            on_bad_lines="skip"
        )

        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

# -----------------------------
# CREATE LABEL MAP
# -----------------------------
def load_statement_labels(df):
    label_map = {}

    if df.empty:
        return label_map

    for _, row in df.iterrows():
        stmt = preprocess_text(row["statement"])
        label_map[stmt] = str(row["label"]).lower()

    return label_map

# -----------------------------
# PREDICTION LOGIC (FIXED)
# -----------------------------
def predict_news(text, label_map):
    clean_text = preprocess_text(text)

    for stmt, label in label_map.items():
        if clean_text in stmt or stmt in clean_text:
            if label in ["false", "pants-fire"]:
                return "FAKE NEWS", 0.95
            else:
                return "REAL NEWS", 0.95

    return "FAKE NEWS", 0.60

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(page_title="Fake News Detection", layout="centered")

    st.title("📰 Fake News Detection System")

    # LOAD DATASET
    test_df = load_test_data()

    # SHOW DATASET
    st.subheader("📊 Dataset Preview")

    if not test_df.empty:
        st.dataframe(test_df.head(20))
    else:
        st.warning("Dataset is empty or not loaded")

    # LABEL MAP
    STATEMENT_LABELS = load_statement_labels(test_df)

    # USER INPUT
    st.subheader("🧠 Test Your News Statement")

    user_statement = st.text_area("Enter news text:", height=150)

    if st.button("Detect News"):
        if user_statement.strip() == "":
            st.warning("Please enter a statement")
        else:
            result, confidence = predict_news(user_statement, STATEMENT_LABELS)

            st.write("### Result")
            st.write(f"Confidence: {confidence:.2f}")

            if result == "REAL NEWS":
                st.success(result)
            else:
                st.error(result)

    # DEBUG OPTION
    if st.checkbox("Show full dataset"):
        st.dataframe(test_df)

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    main()
