import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
def load_data():
    train_df = pd.read_csv('train.tsv', sep='\t', header=None, names=['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    valid_df = pd.read_csv('valid.tsv', sep='\t', header=None, names=['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    test_df = pd.read_csv('test.tsv', sep='\t', header=None, names=['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'])
    
    # Combine train and valid
    df = pd.concat([train_df, valid_df])
    return df, test_df

# Preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Map labels
# Treat only clearly false labels as Fake.
# This helps reduce the number of legitimate statements being classified as Fake.
def map_label(label):
    if label in ['false', 'pants-fire']:
        return 0  # Fake
    return 1  # Real

# Main
if __name__ == "__main__":
    df, test_df = load_data()
    
    # Preprocess
    df['clean_statement'] = df['statement'].apply(preprocess_text)
    test_df['clean_statement'] = test_df['statement'].apply(preprocess_text)
    
    # Map labels
    df['label_binary'] = df['label'].apply(map_label)
    test_df['label_binary'] = test_df['label'].apply(map_label)
    
    # TF-IDF for baseline
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(df['clean_statement'])
    X_test_tfidf = tfidf.transform(test_df['clean_statement'])
    y_train = df['label_binary']
    y_test = test_df['label_binary']
    
    # Baseline: Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)
    print("Logistic Regression:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}")
    print(f"Precision: {precision_score(y_test, y_pred_lr)}")
    print(f"Recall: {recall_score(y_test, y_pred_lr)}")
    print(f"F1: {f1_score(y_test, y_pred_lr)}")
    
    # Advanced: Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_tfidf, y_train)
    y_pred_rf = rf.predict(X_test_tfidf)
    print("\nRandom Forest:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
    print(f"Precision: {precision_score(y_test, y_pred_rf)}")
    print(f"Recall: {recall_score(y_test, y_pred_rf)}")
    print(f"F1: {f1_score(y_test, y_pred_rf)}")
    
    # For LSTM, need sequences
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['clean_statement'])
    X_train_seq = tokenizer.texts_to_sequences(df['clean_statement'])
    X_test_seq = tokenizer.texts_to_sequences(test_df['clean_statement'])
    X_train_pad = pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = pad_sequences(X_test_seq, maxlen=100)
    
    # LSTM Model
    model = Sequential()
    model.add(Embedding(5000, 128, input_length=100))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=1, batch_size=64, validation_split=0.2)
    y_pred_lstm = (model.predict(X_test_pad) > 0.5).astype(int).flatten()
    print("\nLSTM:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lstm)}")
    print(f"Precision: {precision_score(y_test, y_pred_lstm)}")
    print(f"Recall: {recall_score(y_test, y_pred_lstm)}")
    print(f"F1: {f1_score(y_test, y_pred_lstm)}")
    
    # Save models
    pickle.dump(lr, open('lr_model.pkl', 'wb'))
    pickle.dump(rf, open('rf_model.pkl', 'wb'))
    model.save('lstm_model.h5')
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
    pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))