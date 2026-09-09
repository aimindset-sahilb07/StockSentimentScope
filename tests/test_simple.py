"""
Simple tests to validate imports work correctly.
"""
import pytest

def test_imports():
    """Test that all modules can be imported."""
    # Config
    from stock_sentiment_scope import config
    
    # Models
    from stock_sentiment_scope.models.data_models import (
        StockData, Article, SentimentResult, ChatContext
    )
    
    # Services
    from stock_sentiment_scope.services.stock_service import fetch_stock_data
    from stock_sentiment_scope.services.news_service import fetch_news
    from stock_sentiment_scope.services.sentiment_service import (
        analyze_texts, analyze_articles, build_trend_dataframe
    )
    from stock_sentiment_scope.services.chat_service import chat_response
    
    # UI Components
    from stock_sentiment_scope.ui.components import plot_trend
    from stock_sentiment_scope.ui.dashboard import run_app
    
    assert True, "All imports successful"
