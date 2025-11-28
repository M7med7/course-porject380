import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


class SentimentClassifier:
    """Sentiment classifier using TF-IDF and Logistic Regression."""
    
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver='liblinear'
        )
        self.is_trained = False
    
    def fit(self, X_train: list, y_train: np.ndarray) -> 'SentimentClassifier':
        # Transform text to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_tfidf, y_train)
        self.is_trained = True
        
        return self
    
    def predict(self, texts: list) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(X_tfidf)
    
    def predict_proba(self, texts: list) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_tfidf)
    
    def evaluate(self, X_test: list, y_test: np.ndarray) -> dict:
        predictions = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
    
    def save(self, model_path: Path, vectorizer_path: Path) -> None:
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    @classmethod
    def load(cls, model_path: Path, vectorizer_path: Path) -> 'SentimentClassifier':
        classifier = cls()
        classifier.model = joblib.load(model_path)
        classifier.vectorizer = joblib.load(vectorizer_path)
        classifier.is_trained = True
        return classifier


def load_model(model_path: str, vectorizer_path: str) -> Tuple:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer