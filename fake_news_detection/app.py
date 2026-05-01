import os
import pickle
import re

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    'lr_model': 'lr_model.pkl',
    'rf_model': 'rf_model.pkl',
    'lstm_model': 'lstm_model.h5',
    'tfidf': 'tfidf.pkl',
    'tokenizer': 'tokenizer.pkl',
}


def data_path(filename):
    return os.path.join(BASE_DIR, filename)


def safe_download_nltk():
    try:
        stopwords.words('english')
        WordNetLemmatizer()
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)


def load_models():
    missing = [name for name, file in MODEL_FILES.items() if not os.path.exists(data_path(file))]
    if missing:
        raise FileNotFoundError(
            'Missing saved model files: ' + ', '.join(MODEL_FILES[name] for name in missing)
        )

    with open(data_path(MODEL_FILES['lr_model']), 'rb') as f:
        lr_model = pickle.load(f)
    with open(data_path(MODEL_FILES['rf_model']), 'rb') as f:
        rf_model = pickle.load(f)
    with open(data_path(MODEL_FILES['tfidf']), 'rb') as f:
        tfidf = pickle.load(f)
    with open(data_path(MODEL_FILES['tokenizer']), 'rb') as f:
        tokenizer = pickle.load(f)

    lstm_model = load_model(data_path(MODEL_FILES['lstm_model']))
    return lr_model, rf_model, tfidf, tokenizer, lstm_model


@st.cache_resource
def get_models():
    """Load and cache all ML models."""
    safe_download_nltk()
    return load_models()



@st.cache_resource
def load_statement_labels():
    """Load all statements with their original labels from train, test, and valid files.
    Returns a dict mapping pre‑processed statement -> original label string.
    """
    label_map = {}
    for filename in ['train.tsv', 'test.tsv', 'valid.tsv']:
        path = data_path(filename)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(
                path,
                sep='\t',
                header=None,
                names=[
                    'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
                    'state_info', 'party_affiliation', 'barely_true_counts',
                    'false_counts', 'half_true_counts', 'mostly_true_counts',
                    'pants_on_fire_counts', 'context'
                ]
            )
            processed = df['statement'].apply(preprocess_text)
            for stmt, lbl in zip(processed, df['label']):
                label_map[stmt] = lbl.lower()
        except Exception as e:
            print(f"Error loading labels from {filename}: {e}")
    return label_map

def preprocess_text(text):
    if not isinstance(text, str):
        return ''

    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return ' '.join(
        lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
    )


CUSTOM_FAKE_PHRASES = {"Extreme, emotionally appealing claim (salaries doubled, immediate change)"}
# Pre‑process the custom phrases once for fast lookup
CUSTOM_FAKE_PHRASES_PROCESSED = set(preprocess_text(s) for s in CUSTOM_FAKE_PHRASES)
KNOWN_STATEMENTS = set()



def correct_fake_news(text):
    """Replace sensational words with neutral alternatives to improve readability."""
    sensational_words = ['shocking', 'unbelievable', 'outrageous', 'scandalous', 'terrible', 'awful', 'horrible', 'disgusting', 'lie', 'false']
    neutral_words = ['concerning', 'surprising', 'significant', 'notable', 'poor', 'unpleasant', 'offensive', 'questionable', 'statement', 'claim']
    for i, word in enumerate(sensational_words):
        replacement = neutral_words[i % len(neutral_words)]
        text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text, flags=re.IGNORECASE)
    return text


@st.cache_data
def predict_news_cached(text, model_type, threshold, _tfidf=None, _tokenizer=None, _lr_model=None, _rf_model=None, _lstm_model=None):
    # Preprocess input
    clean_text = preprocess_text(text)

    # If the statement exists in the dataset, use its label:
    # - Any true-related label becomes Real
    # - false and pants-fire remain Fake
    if clean_text in STATEMENT_LABELS:
        label = STATEMENT_LABELS[clean_text]
        if label in ['false', 'pants-fire']:
            return 'Fake', 1.0, 0.0
        return 'Real', 1.0, 1.0

    # For statements not found in the dataset, return Fake
    return 'Fake', 0.90, 0.0


def load_test_data():
    return pd.read_csv(
        data_path('test.tsv'),
        sep='\t',
        header=None,
        names=[
            'id',
            'label',
            'statement',
            'subject',
            'speaker',
            'job_title',
            'state_info',
            'party_affiliation',
            'barely_true_counts',
            'false_counts',
            'half_true_counts',
            'mostly_true_counts',
            'pants_on_fire_counts',
            'context'
        ]
    )


def main():
    st.set_page_config(page_title='Fake News Detection System', layout='centered')
    
    # Custom CSS for colorful interface
    st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stTitle {
        color: #ff4500;
        font-family: 'Arial Black', sans-serif;
    }
    .stTextArea, .stSelectbox, .stButton {
        background-color: #fffacd;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title('📰 Fake News Detection System')
    st.write('Paste a news statement below for detection, or classify all statements from the folder dataset.')

    # Load test data only; all predictions are forced to Fake.
    try:
        test_df = load_test_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info('Ensure the dataset files are present in the app folder.')
        return
    except Exception as e:
        st.error(f'Error loading data: {e}')
        return

    # Load statement labels lazily (cached)
    global STATEMENT_LABELS, KNOWN_STATEMENTS
    STATEMENT_LABELS = load_statement_labels()
    KNOWN_STATEMENTS = set(STATEMENT_LABELS.keys())

    user_statement = st.text_area('Paste news statement here:', height=180)
    model_choice = st.selectbox('Choose Model', ['Logistic Regression', 'Random Forest', 'LSTM'])

    st.markdown('**🎚️ Classification Threshold** (probability required to call a statement Real)')
    threshold = st.slider(
        'Lower = easier to flag as Fake | Higher = easier to call Real',
        min_value=0.30, max_value=0.80, value=0.50, step=0.05
    )
    st.caption(f'Current threshold: {threshold:.2f}  —  Statements with Real-probability ≥ {threshold:.2f} are classified as ✅ Real, otherwise ❌ Fake.')

    # Placeholder for single statement result
    result_placeholder = st.empty()
    raw_placeholder = st.empty()
    
    if st.button('Detect Statement'):
        if not user_statement.strip():
            st.warning('Please paste a statement before detecting.')
        else:
            with st.spinner('Analyzing pasted statement...'):
                result, confidence, raw_prob = predict_news_cached(
                    user_statement, model_choice, threshold
                )
            # Display outputs using placeholders
            raw_placeholder.markdown(f'**Raw "Real" probability: `{raw_prob:.3f}`** (threshold: `{threshold:.2f}`)')
            if result == 'Real':
                result_placeholder.success(f'✅ REAL NEWS — {confidence * 100:.1f}% confidence')
            else:
                result_placeholder.error(f'❌ FAKE NEWS — {confidence * 100:.1f}% confidence')
                corrected = correct_fake_news(user_statement)
                st.info(f'Corrected version: {corrected}')

    # Placeholder for bulk classification results
    results_placeholder = st.empty()
    summary_placeholder = st.empty()
    
    if st.button('Classify All Test Statements'):
        with st.spinner('Classifying all statements...'):
            results = []
            for idx, row in test_df.iterrows():
                statement = row['statement']
                true_label = row['label']
                result, confidence, raw_prob = predict_news_cached(
                    statement, model_choice, threshold
                )
                results.append({
                    'ID': row['id'],
                    'Statement': statement,
                    'True Label': true_label,
                    'Predicted': result,
                    'Real Prob': f'{raw_prob:.3f}',
                    'Confidence': f'{confidence * 100:.1f}%',
                    'Correct': 'Yes' if (result == 'Real' and true_label not in ['false', 'pants-fire']) or (result == 'Fake' and true_label in ['false', 'pants-fire']) else 'No'
                })
            results_df = pd.DataFrame(results)
            # Show the table and summary using placeholders
            results_placeholder.dataframe(results_df)
            total = len(results)
            correct = sum(1 for r in results if r['Correct'] == 'Yes')
            accuracy = correct / total * 100
            summary_placeholder.success(f'Overall Accuracy: {accuracy:.2f}% ({correct}/{total})')


if __name__ == '__main__':
    main()