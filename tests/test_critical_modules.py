"""
Tests focused on critical modules and fixes for known issues.
"""
import pytest
import pandas as pd
import numpy as np

# Test Yahoo Finance data handling (addressing Series vs DataFrame issue)
def test_yahoo_finance_data_handling():
    """Test the fix for Yahoo Finance data handling (Series vs DataFrame)."""
    from stock_sentiment_scope.services.sentiment_service import build_trend_dataframe
    
    # Test with normal Series data
    dates = pd.date_range('2025-01-01', periods=5, freq='D')
    stock_data = pd.DataFrame({'Close': [100, 110, 120, 130, 140]}, index=dates)
    scores = [0.1, 0.2, 0.3, 0.4]
    
    # This should work with Series
    df1 = build_trend_dataframe(stock_data, scores)
    assert df1['price'].tolist() == [100, 110, 120, 130]
    
    # Test with DataFrame return type
    df_data = pd.DataFrame(index=dates)
    # Create a case where 'Close' is a DataFrame instead of Series
    close_df = pd.DataFrame([[100], [110], [120], [130], [140]], index=dates)
    df_data['Close'] = close_df
    
    # This should work with DataFrame
    df2 = build_trend_dataframe(df_data, scores)
    assert df2['price'].tolist() == [100, 110, 120, 130]
    
    # Ensure each column is 1D
    for col in df2.columns:
        arr = np.array(df2[col])
        assert arr.ndim == 1, f"Column {col} is not 1D"

# Test Azure OpenAI integration
def test_azure_openai_integration():
    """Test the Azure OpenAI integration fixes."""
    from stock_sentiment_scope.services.chat_service import chat_response
    from stock_sentiment_scope.models.data_models import ChatContext, StockData, Article
    import inspect
    
    # Check the function signature
    sig = inspect.signature(chat_response)
    assert 'context' in sig.parameters
    
    # Check the implementation
    source = inspect.getsource(chat_response)
    
    # Verify using the correct client call pattern
    assert "client.chat.completions.create" in source
    assert "model=deployment_name" in source or "model=" in source
    assert "messages=" in source
    
    # Make sure we're not using the old API pattern
    assert "engine=" not in source
    assert "completion.create" not in source
    
# Test stock service data handling
def test_stock_service_data_handling():
    """Test the stock service data handling with both DataFrame and Series cases."""
    import yfinance as yf
    from stock_sentiment_scope.services.stock_service import fetch_stock_data
    
    # Create test data
    dates = pd.date_range('2025-01-01', periods=3, freq='D')
    
    # Case 1: Normal Series for 'Close'
    df_series = pd.DataFrame({
        'Close': [100, 150, 200],
        'Open': [95, 145, 195]
    }, index=dates)
    
    # Case 2: DataFrame for 'Close'
    df_dataframe = pd.DataFrame({
        'Open': [95, 145, 195]
    }, index=dates)
    df_dataframe['Close'] = pd.DataFrame([[100], [150], [200]], index=dates)
    
    # Monkeypatch yfinance to return our test data
    with pytest.MonkeyPatch().context() as mp:
        # Test with Series case
        mp.setattr(yf, 'download', lambda *args, **kwargs: df_series)
        result_series = fetch_stock_data('TEST', 'Last Week')
        assert result_series is not None
        assert result_series.current_price == 200
        
        # Test with DataFrame case
        mp.setattr(yf, 'download', lambda *args, **kwargs: df_dataframe)
        result_df = fetch_stock_data('TEST', 'Last Week')
        assert result_df is not None
        assert result_df.current_price == 200
