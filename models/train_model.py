import sys
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.preprocessing import load_imdb_dataset, TextPreprocessor
from models.sentiment_model import SentimentClassifier
from utils.config import MODEL_PATH, VECTORIZER_PATH



def main():
    """Main training function."""
    print("=" * 60)
    print("Movie Review Sentiment Analyzer - Model Training")
    print("=" * 60)
    
    print("\n[1/5] Loading IMDB dataset...")
    X_train, y_train, X_test, y_test = load_imdb_dataset(num_words=10000)
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")
    
    print("\n[2/5] Preprocessing training data...")
    preprocessor = TextPreprocessor()
    X_train_clean = preprocessor.preprocess_batch(X_train[:25000])
    print(f"Preprocessed {len(X_train_clean)} training samples")
    
    print("\n[3/5] Preprocessing test data...")
    X_test_clean = preprocessor.preprocess_batch(X_test[:25000])
    print(f"Preprocessed {len(X_test_clean)} test samples")
    
    print("\n[4/5] Training model (this may take a few minutes)...")
    classifier = SentimentClassifier(max_features=5000)
    classifier.fit(X_train_clean, y_train[:25000])
    print("Model training complete!")
    
    print("\n[5/5] Evaluating model...")
    results = classifier.evaluate(X_test_clean, y_test[:25000])
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(results['classification_report'])
    
    print("\n" + "=" * 60)
    print("Confusion Matrix:")
    print("=" * 60)
    print(results['confusion_matrix'])
    
    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)
    classifier.save(MODEL_PATH, VECTORIZER_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    
    print("\n" + "=" * 60)
    print("Model training complete!")
    print("=" * 60)
    print("\nYou can now run the Streamlit app:")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()