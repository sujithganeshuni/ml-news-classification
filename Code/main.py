# main.py
from data_preprocessing import load_data, clean_data, preprocess_data
from model_training import train_model, save_model
from model_evaluation import evaluate_model
from utils import ensure_directory_exists

def main():
    # Paths
    raw_data_path = "data/raw/news_articles.csv"
    processed_data_path = "data/processed/cleaned_news_articles.csv"
    model_save_path = "models/news_classifier.pkl"

    # Ensure directories exist
    ensure_directory_exists("data/processed")
    ensure_directory_exists("models")

    # Load and clean data
    df = load_data(raw_data_path)
    df = clean_data(df)

    # Preprocess data
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df, text_column="text", label_column="category")

    # Train model
    model = train_model(X_train, y_train)

    # Save model
    save_model(model, model_save_path)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()