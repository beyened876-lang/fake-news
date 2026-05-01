# Fake News Detection System

This project implements a Fake News Detection System using NLP and machine learning techniques.

## Dataset

The dataset used is the LIAR dataset from the University of California, Santa Barbara.
- Source: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- Citation: William Yang Wang. "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. arXiv:1705.00648, 2017.
- Size: Approximately 12,800 samples in training, 1,284 in validation, 1,284 in test.

Labels are mapped to binary: 'true' and 'mostly-true' as Real (1), others as Fake (0).

## Preprocessing

- Text cleaning: Remove punctuation, lowercase.
- Stop word removal and lemmatization using NLTK.

## Modeling

### Baseline Model: Logistic Regression
- Justified: Simple, interpretable, good for text classification with TF-IDF features.

### Advanced Model: Random Forest
- Justified: Ensemble method, handles non-linearity, robust to overfitting.

### Neural Network: LSTM
- Justified: Captures sequential dependencies in text, suitable for NLP tasks.

## Evaluation

Metrics: Accuracy, Precision, Recall, F1-score.
- Train/Test split: Using provided train/test sets.

## Deployment

Simple interface using Streamlit.

## Installation

1. Install Python 3.8+.
2. Install dependencies: `pip install -r requirements.txt`

## Usage

Run training first if you need to recreate the saved models:

```bash
cd fake_news_detection
python model_training.py
```

Then launch the app in your browser:

```bash
cd ..
python run_app.py
```

Or run Streamlit directly:

```bash
cd fake_news_detection
streamlit run app.py
```

Enter news text in the app and select a model to classify.
