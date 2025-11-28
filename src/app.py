import streamlit as st
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sentiment_model import SentimentClassifier
from data.preprocessing import TextPreprocessor
from utils.visualization import display_sentiment_result
from utils.config import MODEL_PATH, VECTORIZER_PATH, SENTIMENT_LABELS, COLORS


# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_cached():
    """Load the trained model and return classifier instance."""
    try:
        classifier = SentimentClassifier.load(MODEL_PATH, VECTORIZER_PATH)
        return classifier
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train the model first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def main():
    """Main application function."""
    
    # Title and description
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.markdown("""
    ### Analyze the sentiment of movie reviews using AI
    
    This application uses Natural Language Processing and Machine Learning 
    to determine whether a movie review is **Positive** or **Negative**.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **How it works:**
        1. Enter a movie review
        2. Click 'Analyze Sentiment'
        3. View the predicted sentiment with confidence score
        
        **Model:** Logistic Regression with TF-IDF
        
        **Dataset:** IMDB Movie Reviews (50,000 reviews)
        """)
        
        st.header("📊 Model Performance")
        st.metric("Accuracy", "~88%")
        st.metric("Training Samples", "25,000")
        st.metric("Test Samples", "25,000")
        
        st.header("🔧 Settings")
        preprocess_options = st.checkbox("Advanced Preprocessing", value=True)
        show_probabilities = st.checkbox("Show Probabilities", value=True)
    
    # Load model
    with st.spinner("Loading model..."):
        classifier = load_model_cached()
        preprocessor = TextPreprocessor()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        review_text = st.text_area(
            "Enter your movie review:",
            height=200,
            placeholder="Type or paste a movie review here...\n\n"
                       "Example: This movie was absolutely amazing! "
                       "The acting was superb and the plot kept me engaged "
                       "throughout. Highly recommended!",
            help="Enter the review you want to analyze"
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", 
                                   use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        - Write complete sentences
        - Include specific details
        - Express clear opinions
        - Use natural language
        """)
        
        st.markdown("### 📝 Example Reviews")
        if st.button("Positive Example"):
            st.session_state['example_text'] = (
                "This movie exceeded all my expectations! The cinematography "
                "was breathtaking, the acting was phenomenal, and the story "
                "kept me on the edge of my seat. A masterpiece!"
            )
        
        if st.button("Negative Example"):
            st.session_state['example_text'] = (
                "What a disappointing film. The plot was confusing, the pacing "
                "was terrible, and I found myself checking my watch multiple times. "
                "Would not recommend."
            )
    
    # Use example text if available
    if 'example_text' in st.session_state:
        review_text = st.session_state['example_text']
        del st.session_state['example_text']
        st.rerun()
    
    # Perform prediction
    if analyze_button:
        if not review_text.strip():
            st.warning("⚠️ Please enter a review to analyze.")
        elif len(review_text.split()) < 3:
            st.warning("⚠️ Please enter a longer review (at least 3 words).")
        else:
            with st.spinner("Analyzing sentiment..."):
                try:
                    # Preprocess text
                    if preprocess_options:
                        processed_text = preprocessor.preprocess(review_text)
                    else:
                        processed_text = review_text
                    
                    # Make prediction
                    prediction = classifier.predict([processed_text])[0]
                    probabilities = classifier.predict_proba([processed_text])[0]
                    
                    # Get sentiment label and confidence
                    sentiment = SENTIMENT_LABELS[prediction]
                    confidence = np.max(probabilities)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    st.markdown("---")
                    
                    # Show sentiment result
                    display_sentiment_result(sentiment, confidence, 
                                           probabilities, COLORS)
                    
                    # Additional info
                    with st.expander("📋 Analysis Details"):
                        st.write("**Original Review:**")
                        st.write(review_text)
                        
                        if preprocess_options:
                            st.write("**Preprocessed Text:**")
                            st.write(processed_text)
                        
                        st.write("**Prediction:**", prediction)
                        
                        if show_probabilities:
                            st.write("**Probability Scores:**")
                            prob_df = {
                                "Sentiment": ["Negative", "Positive"],
                                "Probability": [f"{p:.2%}" for p in probabilities]
                            }
                            st.table(prob_df)
                
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Made with ❤️ using Streamlit and Scikit-learn</p>
        <p>Movie Review Sentiment Analyzer | 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()