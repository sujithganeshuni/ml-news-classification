# data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    """Load the dataset from a file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset (remove missing values, duplicates, etc.)."""
    df = df.dropna()  # Remove missing values
    df = df.drop_duplicates()  # Remove duplicates
    return df

def preprocess_data(df, text_column, label_column):
    """Preprocess the data (split into features and labels, vectorize text)."""
    # Split into features and labels
    X = df[text_column]
    y = df[label_column]

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer