# model_training.py
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")