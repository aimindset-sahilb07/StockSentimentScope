"""
Data models for StockSentimentScope application.
Contains typing definitions for data structures used across services.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

@dataclass
class StockData:
    """Stock price data with metadata"""
    data: pd.DataFrame
    current_price: float
    change_pct: float
    ticker: str = ""

@dataclass
class Article:
    """News article with metadata and sentiment analysis"""
    title: str
    url: str
    description: Optional[str] = None
    content: Optional[str] = None
    source_name: Optional[str] = None
    published_at: Optional[datetime] = None
    sentiment_label: str = "neutral"  # positive, neutral, negative
    sentiment_score: float = 0.0
    
@dataclass
class SentimentResult:
    """Result of sentiment analysis on a text"""
    score: float  # Range typically -1 to 1
    label: str    # positive, neutral, negative
    confidence: float = 1.0  # Range 0 to 1
    
@dataclass
class ChatContext:
    """Context information for chat responses"""
    stock: StockData
    articles: List[Article]
    overall_sentiment: float
