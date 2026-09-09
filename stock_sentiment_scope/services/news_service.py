"""
News service for StockSentimentScope.
Handles fetching and processing news articles from NewsAPI.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

from stock_sentiment_scope.config import get_newsapi_client, NEWS_API_AVAILABLE
from stock_sentiment_scope.models.data_models import Article
from stock_sentiment_scope.utils.logging import get_logger
from stock_sentiment_scope.utils.cache import cached_data

logger = get_logger(__name__)

@cached_data(ttl_seconds=600)  # Cache for 10 minutes
def fetch_news(ticker: str, time_frame: str) -> List[Article]:
    """
    Fetch news articles from NewsAPI based on ticker and time frame.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        time_frame: Time frame string ('Last 24 Hours', 'Last 3 Days', 'Last Week', 'Last Month')
        
    Returns:
        List of Article objects
    """
    newsapi = get_newsapi_client()
    
    if not newsapi or not NEWS_API_AVAILABLE:
        logger.warning("News API is not available. Unable to fetch news articles.")
        st.warning("News API is not available. Unable to fetch news articles.")
        return []
    
    # Use timezone-aware datetime object (fix deprecation warning)
    to_date = datetime.now(timezone.utc)
    
    # Determine from_date based on time_frame
    if time_frame == 'Last 24 Hours':
        from_date = to_date - timedelta(days=1)
    elif time_frame == 'Last 3 Days':
        from_date = to_date - timedelta(days=3)
    elif time_frame == 'Last Week':
        from_date = to_date - timedelta(days=7)
    else:  # Last Month
        from_date = to_date - timedelta(days=30)
    
    try:
        # Format dates correctly as YYYY-MM-DD
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching news for {ticker} from {from_date_str} to {to_date_str}")
        
        # Query NewsAPI
        response = newsapi.get_everything(
            q=ticker, 
            from_param=from_date_str, 
            to=to_date_str, 
            language='en', 
            sort_by='relevancy', 
            page_size=20
        )
        
        # Convert raw articles to our Article model
        articles = []
        for raw_article in response.get('articles', []):
            try:
                # Parse the published date
                published_at = None
                if raw_article.get('publishedAt'):
                    try:
                        published_at = pd.to_datetime(raw_article['publishedAt'])
                    except:
                        pass
                
                # Create an Article object
                article = Article(
                    title=raw_article.get('title', ''),
                    url=raw_article.get('url', ''),
                    description=raw_article.get('description'),
                    content=raw_article.get('content'),
                    source_name=raw_article.get('source', {}).get('name'),
                    published_at=published_at
                )
                articles.append(article)
                
            except Exception as e:
                logger.warning(f"Error processing article: {str(e)}")
                continue
                
        return articles
        
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        logger.error(f"NewsAPI error: {str(e)}")
        return []
