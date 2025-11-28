import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from typing import Dict
import streamlit as st


def plot_confidence_gauge(confidence: float, sentiment: str, colors: Dict[str, str]) -> go.Figure:

    color = colors.get(sentiment.lower(), colors.get("neutral"))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FFE5E5'},
                {'range': [50, 75], 'color': '#FFF9E5'},
                {'range': [75, 100], 'color': '#E5FFE5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'size': 16}
    )
    
    return fig


def plot_probability_bar(probabilities: np.ndarray, labels: list,colors: Dict[str, str]) -> go.Figure:

    bar_colors = [colors.get(label.lower(), colors.get("neutral")) 
                  for label in labels]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities * 100,
            marker_color=bar_colors,
            text=[f'{p:.1f}%' for p in probabilities * 100],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Probability Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Probability (%)",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 100]),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig


def display_sentiment_result(sentiment: str, confidence: float,probabilities: np.ndarray, colors: Dict[str, str]) -> None:

    # Display sentiment with colored background
    sentiment_color = colors.get(sentiment.lower(), colors.get("neutral"))
    
    st.markdown(
        f"""
        <div style='padding: 20px; border-radius: 10px; 
                    background-color: {sentiment_color}20; 
                    border-left: 5px solid {sentiment_color};'>
            <h2 style='color: {sentiment_color}; margin: 0;'>
                Sentiment: {sentiment}
            </h2>
            <p style='font-size: 18px; margin-top: 10px;'>
                Confidence: {confidence:.2%}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display gauge chart
    st.plotly_chart(
        plot_confidence_gauge(confidence, sentiment, colors),
        use_container_width=True
    )
    
    # Display probability distribution
    labels = ["Negative", "Positive"]
    st.plotly_chart(
        plot_probability_bar(probabilities, labels, colors),
        use_container_width=True
    )