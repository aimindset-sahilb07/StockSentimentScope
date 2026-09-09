"""
Configuration module for StockSentimentScope.
Handles loading environment variables and initializing API clients.
"""
import os
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from newsapi import NewsApiClient

from stock_sentiment_scope.utils.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Configuration flags
OPENAI_AVAILABLE = False
NEWS_API_AVAILABLE = False
VADER_AVAILABLE = False
FINBERT_AVAILABLE = False

# Default values
DEFAULT_API_VERSION = "2024-02-15-preview"

# Azure OpenAI client
try:
    from openai import AzureOpenAI  # Using new SDK format
    
    # Get configuration values from environment
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "stocksentiment-gpt-4o")
    
    # Initialize client only if we have required values
    if api_key and endpoint:
        openai_client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        OPENAI_AVAILABLE = True
        logger.info(f"Azure OpenAI client initialized successfully with API version {api_version}")
    else:
        logger.warning("Azure OpenAI client not initialized: missing API key or endpoint")
except (ImportError, Exception) as e:
    logger.warning(f"Azure OpenAI client initialization failed: {str(e)}")
    openai_client = None

# NewsAPI client
try:
    news_api_key = os.getenv("NEWS_API_KEY")
    if news_api_key:
        newsapi_client = NewsApiClient(api_key=news_api_key)
        NEWS_API_AVAILABLE = True
        logger.info("NewsAPI client initialized successfully")
    else:
        logger.warning("NewsAPI client not initialized: missing API key")
        newsapi_client = None
except Exception as e:
    logger.warning(f"NewsAPI client initialization failed: {str(e)}")
    newsapi_client = None

# NLTK VADER
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    nltk.download('vader_lexicon', quiet=True)
    vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
    logger.info("VADER sentiment analyzer initialized successfully")
except Exception as e:
    logger.warning(f"VADER initialization failed: {str(e)}")
    vader_analyzer = None

def get_openai_client() -> Optional[AzureOpenAI]:
    """Get the Azure OpenAI client instance"""
    return openai_client

def get_newsapi_client() -> Optional[NewsApiClient]:
    """Get the NewsAPI client instance"""
    return newsapi_client

def get_vader_analyzer() -> Optional[SentimentIntensityAnalyzer]:
    """Get the VADER sentiment analyzer instance"""
    return vader_analyzer

def get_azure_deployment_name() -> str:
    """Get the Azure OpenAI deployment name"""
    return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "stocksentiment-gpt-4o")
