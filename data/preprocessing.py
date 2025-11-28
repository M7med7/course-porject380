import re
import string
import pandas as pd
import numpy as np
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:

    def __init__(self):
        """Initialize the preprocessor with stopwords and lemmatizer."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def preprocess(self, text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
        text = self.clean_text(text)
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        if lemmatize:
            text = self.lemmatize_text(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str], remove_stopwords: bool = True,lemmatize: bool = True) -> List[str]:
        return [self.preprocess(text, remove_stopwords, lemmatize) for text in texts]


def load_imdb_dataset(num_words: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load IMDB dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
    
    # Get word index
    word_index = imdb.get_word_index()
    
    # Reverse word index to decode reviews
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    # Decode reviews
    def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    
    X_train_decoded = [decode_review(review) for review in X_train]
    X_test_decoded = [decode_review(review) for review in X_test]
    
    return X_train_decoded, y_train, X_test_decoded, y_test


def create_dataframe(texts: List[str], labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        'review': texts,
        'sentiment': labels
    })


def split_dataset(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,random_state: int = 42) -> Tuple:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)