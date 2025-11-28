# 🎬 Movie Review Sentiment Analyzer

An AI-powered sentiment analysis application that classifies movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP) and Machine Learning techniques.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-Educational-green.svg)

## 📋 Overview

This project utilizes advanced NLP techniques and machine learning algorithms to analyze the sentiment of movie reviews. Built with Streamlit, it provides an interactive and user-friendly interface for real-time sentiment prediction with confidence scores and beautiful visualizations.

## ✨ Features

- 🤖 **Real-time Sentiment Analysis** - Instant classification of movie reviews
- 📊 **Interactive Visualizations** - Confidence gauge and probability distribution charts
- 🧹 **Advanced Text Preprocessing** - NLTK-based text cleaning, stopword removal, and lemmatization
- 🎨 **Beautiful UI** - Modern Streamlit interface with responsive design
- 📈 **High Accuracy** - ~88% accuracy on IMDB dataset
- ⚙️ **Customizable Settings** - Toggle preprocessing options and probability display
- 💾 **Trained Models** - Pre-trained models ready for deployment

## 🛠️ Technologies & Libraries

### Core Technologies

- **Python 3.11+** - Programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning (Logistic Regression + TF-IDF)
- **NLTK** - Natural language processing and text preprocessing
- **TensorFlow/Keras** - IMDB dataset loading

### Visualization & Data Processing

- **Plotly** - Interactive charts and gauges
- **Matplotlib & Seaborn** - Statistical visualizations
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations

## 📁 Project Structure

```
movie-review-sentiment-analyzer/
├── src/
│   └── app.py                      # Main Streamlit application
├── data/
│   ├── __init__.py
│   └── preprocessing.py            # Text preprocessing utilities
├── models/
│   ├── __init__.py
│   ├── sentiment_model.py          # ML model implementation
│   └── train_model.py              # Model training script
├── utils/
│   ├── __init__.py
│   ├── config.py                   # Configuration settings
│   ├── visualization.py            # Plotting and visualization functions
│   └── saved_models/               # Trained model files
│       ├── sentiment_model.pkl     # Trained classifier
│       └── vectorizer.pkl          # TF-IDF vectorizer
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-review-sentiment-analyzer.git
cd movie-review-sentiment-analyzer
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

The required NLTK data (stopwords, wordnet) will be downloaded automatically on first run.

### Step 5: Train the Model

```bash
# Run from project root
python -m models.train_model
```

**Training Process:**

1. Downloads IMDB dataset (50,000 reviews)
2. Preprocesses text data (cleaning, stopword removal, lemmatization)
3. Trains Logistic Regression model with TF-IDF features
4. Evaluates model performance
5. Saves trained model files to `utils/saved_models/`

**Training time:** ~5-10 minutes (depending on your system)

## 💻 Usage

### Running the Application

```bash
# Make sure you're in the project root directory
streamlit run src/app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Enter a Movie Review**

   - Type or paste your review in the text area
   - Or click example buttons for pre-filled reviews

2. **Configure Settings** (Optional)

   - Toggle "Advanced Preprocessing" to clean text
   - Toggle "Show Probabilities" to view detailed scores

3. **Analyze Sentiment**

   - Click the "🔍 Analyze Sentiment" button
   - View the predicted sentiment (Positive/Negative)
   - Check the confidence score and probability distribution

4. **Explore Results**
   - Interactive confidence gauge chart
   - Probability distribution bar chart
   - Expandable section for detailed analysis

### Example Reviews

**Positive Review:**

```
This movie exceeded all my expectations! The cinematography was breathtaking,
the acting was phenomenal, and the story kept me on the edge of my seat.
A masterpiece!
```

**Negative Review:**

```
What a disappointing film. The plot was confusing, the pacing was terrible,
and I found myself checking my watch multiple times. Would not recommend.
```

## 📊 Model Performance

### Training Details

- **Dataset:** IMDB Movie Reviews
- **Training Samples:** 25,000 reviews
- **Test Samples:** 25,000 reviews
- **Algorithm:** Logistic Regression
- **Feature Extraction:** TF-IDF (5,000 features, unigrams + bigrams)
- **Accuracy:** ~88.28%

### Classification Report

```
              precision    recall  f1-score   support

    Negative       0.89      0.88      0.88     12500
    Positive       0.88      0.89      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000
```

### Confusion Matrix

```
[[10976  1524]
 [ 1406 11094]]
```

## 🔧 Configuration

### Model Parameters (config.py)

```python
MAX_FEATURES = 5000              # TF-IDF max features
MAX_SEQUENCE_LENGTH = 200        # Max review length
EMBEDDING_DIM = 128              # Embedding dimension
TEST_SIZE = 0.2                  # Train/test split ratio
RANDOM_STATE = 42                # Reproducibility seed
```

### TF-IDF Vectorizer Settings

```python
max_features=5000                # Top 5000 important words
ngram_range=(1, 2)               # Unigrams and bigrams
min_df=2                         # Min document frequency
max_df=0.8                       # Max document frequency
```

### Logistic Regression Settings

```python
max_iter=1000                    # Maximum iterations
random_state=42                  # Random seed
C=1.0                            # Regularization strength
solver='liblinear'               # Optimization algorithm
```

## 🧪 Testing

### Manual Testing

Run the application and test with various inputs:

- Short reviews (< 3 words) - should show warning
- Empty reviews - should show warning
- Positive reviews - should classify correctly
- Negative reviews - should classify correctly
- Mixed sentiment - check confidence scores

### Model Evaluation

```bash
python -m models.train_model
```

View detailed metrics:

- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix

## 🎓 Academic Context

**Course:** CPIT-380 - Multimedia Technologies
**Institution:** King Abdulaziz University (KAU)  
**Faculty:** Faculty of Computing and Information Technology (FCIT)  
**Semester:** 8th Semester, Fall 2025

### Team Members

- Mohammed Alharbi
- Abdulaziz Almutairi
- Fahad Alhawas

## 📚 Key Components Explained

### 1. Text Preprocessing (`data/preprocessing.py`)

- **Text Cleaning:** Lowercase conversion, URL/HTML removal, punctuation/number removal
- **Stopword Removal:** Filters common words (the, is, and, etc.)
- **Lemmatization:** Converts words to base form (loved → love)
- **Batch Processing:** Efficient processing of multiple reviews

### 2. Sentiment Model (`models/sentiment_model.py`)

- **SentimentClassifier Class:** Main model wrapper
- **TF-IDF Vectorization:** Converts text to numerical features
- **Logistic Regression:** Binary classification algorithm
- **Model Persistence:** Save/load trained models

### 3. Visualization (`utils/visualization.py`)

- **Confidence Gauge:** Interactive gauge chart showing prediction confidence
- **Probability Bar Chart:** Distribution of positive/negative probabilities
- **Colored Results:** Dynamic color-coding based on sentiment
- **Confusion Matrix:** Model performance visualization

### 4. Streamlit App (`src/app.py`)

- **Interactive UI:** Text input, buttons, checkboxes
- **Model Caching:** Efficient model loading with `@st.cache_resource`
- **Real-time Prediction:** Instant sentiment analysis
- **Responsive Design:** Two-column layout with sidebar

## 🔮 Future Improvements

### Model Enhancements

- [ ] Implement deep learning models (LSTM, GRU, Transformer)
- [ ] Use pre-trained embeddings (Word2Vec, GloVe)
- [ ] Fine-tune BERT for sentiment analysis
- [ ] Multi-class sentiment (Very Negative, Negative, Neutral, Positive, Very Positive)

### Feature Additions

- [ ] Batch review analysis (upload CSV files)
- [ ] Sentiment history and trends
- [ ] Word cloud visualization for reviews
- [ ] Export results to PDF/CSV
- [ ] Multi-language support
- [ ] API endpoint for integration

### UI/UX Improvements

- [ ] Dark mode theme
- [ ] Custom color themes
- [ ] Review history tracking
- [ ] Comparison mode (multiple reviews)
- [ ] Mobile-responsive design optimization

### Technical Improvements

- [ ] Add comprehensive unit tests
- [ ] Implement CI/CD pipeline
- [ ] Docker containerization
- [ ] Database integration for storing results
- [ ] Performance optimization for large batches

## 🐛 Known Issues

- Model files (`.pkl`) are not included in Git due to size (use `.gitignore`)
- First-time NLTK data download may take a few moments
- Large batch processing may be slow (optimize with multiprocessing)

## 🤝 Contributing

This is an educational project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is created for educational purposes as part of the CPIT-380 course at King Abdulaziz University.

## 🙏 Acknowledgments

- **IMDB Dataset** - Movie review dataset from Keras
- **Streamlit** - For the amazing web framework
- **Scikit-learn** - For machine learning tools
- **NLTK** - For NLP utilities
- **King Abdulaziz University** - For the learning opportunity
- **Dr. [Siam]** - Course instructor and mentor

## 📧 Contact

For questions or feedback:

- **Email:** [Mohammed266433@gmail.com]
- **GitHub Issues:** [Report bugs or request features]
- **University:** King Abdulaziz University, Jeddah, Saudi Arabia

---

<div align="center">

### ⭐ If you find this project helpful, please give it a star!

Made with ❤️ by KAU FCIT Students

**Movie Review Sentiment Analyzer** | CPIT-380 | Fall 2025

</div>
