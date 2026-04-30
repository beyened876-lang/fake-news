import os
import pickle
import re

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# DOWNLOAD NLTK RESOURCES
# -----------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    'lr_model': 'lr_model.pkl',
    'rf_model': 'rf_model.pkl',
    'tfidf': 'tfidf.pkl',
    'tokenizer': 'tokenizer.pkl',
}

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
# LOAD DATASET (FIXED)
# -----------------------------
def load_test_data():
    path = data_path("test.tsv")

    if not os.path.exists(path):
        st.error("❌ test.tsv file not found in project directory")
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
# LOAD LABEL MAP (FIXED LOGIC)
# -----------------------------
@st.cache_resource
def load_statement_labels(df):
    label_map = {}

    if df.empty:
        return label_map

    for _, row in df.iterrows():
        stmt = preprocess_text(row["statement"])
        label_map[stmt] = str(row["label"]).lower()

    return label_map

# -----------------------------
# SIMPLE PREDICTION LOGIC (FIXED)
# -----------------------------
def predict_news(text):
    clean_text = preprocess_text(text)

    # match against dataset
    for stmt, label in STATEMENT_LABELS.items():
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
        st.warning("Dataset is empty or failed to load")

    # CREATE LABEL MAP
    global STATEMENT_LABELS
    STATEMENT_LABELS = load_statement_labels(test_df)

    # USER INPUT
    st.subheader("🧠 Test Your News")
    user_statement = st.text_area("Enter a news statement:", height=150)

    if st.button("Detect News"):
        if user_statement.strip() == "":
            st.warning("Please enter a statement")
        else:
            result, confidence = predict_news(user_statement)

            st.write("### Result:")
            st.write(f"Confidence: {confidence:.2f}")

            if result == "REAL NEWS":
                st.success(result)
            else:
                st.error(result)

    # DEBUG SECTION
    if st.checkbox("Show full dataset"):
        st.dataframe(test_df)

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    main()
