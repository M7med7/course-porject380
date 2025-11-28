# Movie Review Sentiment Analyzer

This project is a Movie Review Sentiment Analyzer that utilizes Natural Language Processing (NLP) and machine learning techniques to analyze the sentiment of movie reviews. The application is built using Streamlit, providing an interactive user interface for users to input their reviews and receive sentiment predictions.

## Project Structure

```
movie-review-sentiment-analyzer
├── src
│   ├── app.py
│   ├── models
│   │   ├── __init__.py
│   │   └── sentiment_model.py
│   ├── data
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── text_processing.py
│   └── config.py
├── notebooks
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
├── tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd movie-review-sentiment-analyzer
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run src/app.py
```

2. Open the provided URL in your web browser.
3. Enter a movie review in the text area and click the "Predict" button.
4. The application will display the predicted sentiment along with the confidence score.

## Project Components

- **Data Preprocessing**: The `src/data/preprocessing.py` file handles the cleaning and preparation of the dataset for training.
- **Model Training**: The `src/models/sentiment_model.py` file contains functions for training the sentiment analysis model using TF-IDF and Logistic Regression.
- **Streamlit App**: The `src/app.py` file serves as the main entry point for the application, providing the user interface for input and displaying predictions.
- **Notebooks**: The `notebooks` directory contains Jupyter notebooks for data exploration and model training.
- **Testing**: The `tests` directory includes unit tests to ensure the functionality of preprocessing and model prediction.

## Future Improvements

- Experiment with advanced models such as LSTM or BERT for improved accuracy.
- Enhance text preprocessing techniques to include stemming and lemmatization.
- Add more visualizations to provide insights into the model's performance.
- Implement user authentication for personalized experiences.

## Acknowledgments

This project leverages various libraries and frameworks, including Streamlit, scikit-learn, pandas, and numpy, to build a robust sentiment analysis application.
