"""
Sentiment analysis service for StockSentimentScope.
Handles analyzing sentiment of news articles.
"""
from typing import List, Dict, Tuple, Any, Optional

import streamlit as st
import numpy as np

from stock_sentiment_scope.config import get_vader_analyzer, VADER_AVAILABLE
from stock_sentiment_scope.models.data_models import SentimentResult, Article
from stock_sentiment_scope.utils.logging import get_logger
from stock_sentiment_scope.utils.cache import cached_data

logger = get_logger(__name__)

@cached_data(ttl_seconds=3600)  # Cache for 1 hour
def analyze_texts(texts: List[str]) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Analyze sentiment of a list of texts.
    
    Args:
        texts: List of text strings to analyze
        
    Returns:
        Tuple of (raw scores, sentiment labels)
        - raw_scores: List of floats between -1 and 1
        - sentiment_labels: List of dicts with 'label' and 'score' keys
    """
    sia = get_vader_analyzer()
    
    if not sia or not VADER_AVAILABLE:
        logger.warning("VADER sentiment analysis is not available.")
        st.warning("VADER sentiment analysis is not available.")
        # Return neutral sentiment for all texts
        return [0.0 for _ in texts], [{'label': 'neutral', 'score': 0.0} for _ in texts]
    
    try:
        # Use VADER for sentiment scores
        scores = [sia.polarity_scores(t)['compound'] for t in texts]
        
        # Convert to label format
        labels = [
            {'label': 'positive' if s > 0.05 else 'negative' if s < -0.05 else 'neutral', 
             'score': abs(s)} 
            for s in scores
        ]
        
        return scores, labels
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        # Return neutral sentiment on error
        return [0.0 for _ in texts], [{'label': 'neutral', 'score': 0.0} for _ in texts]

def analyze_articles(articles: List[Article]) -> List[Article]:
    """
    Enrich articles with sentiment analysis.
    
    Args:
        articles: List of Article objects to analyze
        
    Returns:
        The same list of articles, but with sentiment_label and sentiment_score populated
    """
    # Extract text content from articles
    texts = [a.description or a.title for a in articles if a.description or a.title]
    
    if not texts:
        logger.warning("No text content found in articles to analyze")
        return articles
        
    # Analyze sentiment
    scores, labels = analyze_texts(texts)
    
    # Update articles with sentiment results
    for i, (score, label_info) in enumerate(zip(scores, labels)):
        if i < len(articles):
            articles[i].sentiment_score = score
            articles[i].sentiment_label = label_info['label']
    
    return articles

def build_trend_dataframe(stock_data, vad_scores, max_points: Optional[int] = None):
    """
    Build time-series DataFrame for sentiment trend chart.
    
    Args:
        stock_data: DataFrame with stock price data
        vad_scores: List of sentiment scores
        max_points: Maximum number of data points to include (default: None, use all available)
        
    Returns:
        DataFrame with time, text_score, and price columns
    """
    import pandas as pd
    
    # Determine how many data points to use
    if max_points is None:
        score_len = min(len(vad_scores), len(stock_data))
    else:
        score_len = min(len(vad_scores), len(stock_data), max_points)
    
    # Handle both Series and DataFrame cases for Close prices
    close_data = stock_data['Close']
    if isinstance(close_data, pd.DataFrame):
        # If 'Close' is a DataFrame (happens with some Yahoo API responses)
        price_values = close_data.iloc[:score_len].values.flatten().tolist()
    else:
        # If 'Close' is a Series (standard case)
        price_values = close_data.iloc[:score_len].tolist()
        
    # Create DataFrame with guaranteed 1D columns
    return pd.DataFrame({
        'time': list(stock_data.index[:score_len]),
        'text_score': vad_scores[:score_len],
        'price': price_values
    })

def get_sentiment_summary(articles: List[Article]) -> Dict[str, Any]:
    """
    Generate a summary of sentiment analysis results.
    
    Args:
        articles: List of analyzed Articles
        
    Returns:
        Dictionary with sentiment summary statistics
    """
    if not articles:
        return {
            'avg_sentiment': 0.0,
            'sentiment_label': 'neutral',
            'sentiment_counts': {'positive': 0, 'negative': 0, 'neutral': 0},
            'dominant_sentiment': 'neutral',
            'dominant_count': 0,
            'total_count': 0
        }
    
    # Extract sentiment scores and labels
    scores = [a.sentiment_score for a in articles]
    labels = [a.sentiment_label for a in articles]
    
    # Calculate average sentiment
    avg_sentiment = np.mean(scores) if scores else 0.0
    
    # Determine overall sentiment label
    sentiment_label = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
    
    # Count occurrences of each sentiment label
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for label in labels:
        sentiment_counts[label] += 1
    
    # Determine dominant sentiment
    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    
    return {
        'avg_sentiment': avg_sentiment,
        'sentiment_label': sentiment_label,
        'sentiment_counts': sentiment_counts,
        'dominant_sentiment': dominant_sentiment,
        'dominant_count': sentiment_counts[dominant_sentiment],
        'total_count': len(articles)
    }
