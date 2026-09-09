"""
Stock data service for StockSentimentScope.
Handles fetching and processing stock data from Yahoo Finance.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Union

import pandas as pd
import yfinance as yf

from stock_sentiment_scope.models.data_models import StockData
from stock_sentiment_scope.utils.logging import get_logger
from stock_sentiment_scope.utils.cache import cached_data

logger = get_logger(__name__)

@cached_data(ttl_seconds=300)  # Cache for 5 minutes
def fetch_stock_data(ticker: str, time_frame: str) -> Optional[StockData]:
    """
    Fetch stock data from Yahoo Finance based on ticker and time frame.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        time_frame: Time frame string ('Last 24 Hours', 'Last 3 Days', 'Last Week', 'Last Month')
        
    Returns:
        StockData object or None if fetching fails
    """
    end = datetime.now()
    
    # Determine start date and interval based on time frame
    if time_frame == 'Last 24 Hours':
        start = end - timedelta(days=1)
        interval = '1h'
    elif time_frame == 'Last 3 Days':
        start = end - timedelta(days=3)
        interval = '1h'
    elif time_frame == 'Last Week':
        start = end - timedelta(days=7)
        interval = '1d'
    else:  # Last Month
        start = end - timedelta(days=30)
        interval = '1d'
    
    # Initialize variables
    df = None
    retries = 3
    
    # Attempt to fetch data with retries
    for i in range(retries):
        try:
            logger.info(f"Fetching {ticker} data from {start} to {end} (attempt {i+1}/{retries})")
            
            # Pass auto_adjust explicitly since default has changed
            df = yf.download(
                ticker, 
                start=start, 
                end=end, 
                interval=interval, 
                progress=False, 
                ignore_tz=True,
                auto_adjust=True
            )
            
            if not df.empty:
                break
                
            logger.warning(f"Empty dataframe returned for {ticker}. Retrying...")
            
        except Exception as e:
            logger.warning(f"Attempt {i+1}/{retries} to fetch stock data failed: {str(e)}")
            if i < retries - 1:  # Don't sleep after the last attempt
                time.sleep(2)
    
    # Return None if fetching failed
    if df is None or df.empty:
        logger.error(f"Failed to fetch stock data for {ticker} after {retries} attempts")
        return None
    
    # Handle different return types from Yahoo Finance API
    try:
        # Use .item() to extract scalar floats from the Series without float() call
        # Handle both DataFrame and Series return types for 'Close'
        close_data = df['Close']
        
        # Debugging information
        logger.info(f"Close data type: {type(close_data)}")
        
        if isinstance(close_data, pd.DataFrame):
            # Handle DataFrame case (sometimes returned by Yahoo Finance)
            # Extract values from first column of the DataFrame
            logger.info(f"Close data shape: {close_data.shape}")
            if close_data.shape[1] >= 1:
                # Get values from the first column
                current_price = close_data.iloc[-1, 0]
                previous_price = close_data.iloc[0, 0]
            else:
                # Fallback to flattened values if column structure is unclear
                current_price = close_data.iloc[-1].values.flatten()[0]
                previous_price = close_data.iloc[0].values.flatten()[0]
        else:
            # Handle Series case (standard return type)
            current_price = close_data.iloc[-1]
            previous_price = close_data.iloc[0]
            
        # Calculate percentage change
        change_pct = ((current_price - previous_price) / previous_price * 100)
        
        # Create and return StockData object
        return StockData(
            data=df,
            current_price=current_price,
            change_pct=change_pct,
            ticker=ticker
        )
        
    except Exception as e:
        logger.error(f"Error processing stock data: {str(e)}")
        return None
