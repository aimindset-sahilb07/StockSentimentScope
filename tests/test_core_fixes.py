"""
Tests specifically focused on validating fixes for known issues.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Yahoo Finance data handling test
def test_yahoo_finance_data_handling():
    """Test the critical DataFrame vs Series handling for Yahoo Finance data."""
    from stock_sentiment_scope.services.sentiment_service import build_trend_dataframe
    
    # Create a case with Series
    idx = pd.date_range('2025-01-01', periods=3, freq='D')
    series_df = pd.DataFrame({'Close': [100, 150, 200]}, index=idx)
    scores = [0.1, 0.2, 0.3]
    
    # Should work with Series
    result_series = build_trend_dataframe(series_df, scores)
    assert result_series['price'].tolist() == [100, 150, 200]
    
    # Create a case with DataFrame
    df_case = pd.DataFrame(index=idx)
    df_case['Close'] = pd.DataFrame([[110], [160], [210]], index=idx)
    
    # Should work with DataFrame
    result_df = build_trend_dataframe(df_case, scores)
    assert result_df['price'].tolist() == [110, 160, 210]
    
    print("✓ Yahoo Finance data handling works correctly")
    return True

# Azure OpenAI integration test
def test_azure_openai_integration():
    """Test the Azure OpenAI client integration patterns."""
    from stock_sentiment_scope.config import get_azure_deployment_name
    import inspect
    from stock_sentiment_scope.services.chat_service import chat_response
    
    # Check deployment name accessor
    deployment_name = get_azure_deployment_name()
    assert isinstance(deployment_name, str)
    assert len(deployment_name) > 0
    
    # Check chat response function signature and code
    chat_code = inspect.getsource(chat_response)
    
    # Verify using the correct client call pattern
    assert "client.chat.completions.create" in chat_code
    assert "model=" in chat_code
    assert "messages=" in chat_code
    
    # Make sure we're not using the old API pattern
    assert "engine=" not in chat_code
    assert "completion.create" not in chat_code
    
    print("✓ Azure OpenAI integration uses correct patterns")
    return True

# 1D arrays for DataFrame test
def test_dataframe_1d_arrays():
    """Test that we're creating DataFrames with 1D arrays to avoid errors."""
    from stock_sentiment_scope.services.sentiment_service import build_trend_dataframe
    
    # Create test data
    idx = pd.date_range('2025-01-01', periods=3, freq='D')
    df = pd.DataFrame({'Close': [100, 150, 200]}, index=idx)
    scores = [0.1, 0.2, 0.3]
    
    # Build the DataFrame
    result = build_trend_dataframe(df, scores)
    
    # Check that each column is a 1D array
    for col in result.columns:
        # Get array representation
        arr = np.array(result[col])
        # Check it's 1D
        assert arr.ndim == 1, f"Column {col} is not 1D"
    
    print("✓ DataFrame columns are properly 1D")
    return True

# Integration test for end-to-end workflow
def test_end_to_end_workflow():
    """Test the core workflow from stock data to sentiment analysis."""
    import pandas as pd
    import numpy as np
    from stock_sentiment_scope.models.data_models import Article, StockData
    from stock_sentiment_scope.services.sentiment_service import analyze_articles, build_trend_dataframe
    
    # Create fake stock data
    idx = pd.date_range('2025-01-01', periods=5, freq='D')
    stock_data = StockData(
        data=pd.DataFrame({'Close': [100, 110, 105, 115, 120]}, index=idx),
        current_price=120,
        change_pct=20.0,
        ticker='AAPL'
    )
    
    # Create fake articles
    articles = [
        Article(
            title="Positive news about AAPL",
            url="https://example.com/1",
            description="Apple reported strong earnings",
            published_at=datetime(2025, 1, 1)
        ),
        Article(
            title="Mixed news about AAPL",
            url="https://example.com/2",
            description="Apple faces supply chain challenges",
            published_at=datetime(2025, 1, 2)
        ),
        Article(
            title="Negative news about AAPL",
            url="https://example.com/3",
            description="Apple sued for patent infringement",
            published_at=datetime(2025, 1, 3)
        )
    ]
    
    # Mock the sentiment analysis
    def mock_analyze_texts(texts):
        # Return predictable sentiment scores
        scores = [0.5, 0.0, -0.5]
        labels = [
            {'label': 'positive', 'score': 0.5},
            {'label': 'neutral', 'score': 0.0},
            {'label': 'negative', 'score': 0.5}
        ]
        return scores, labels
    
    # Use pytest's monkeypatch in a with-block style
    with pytest.MonkeyPatch().context() as mp:
        import stock_sentiment_scope.services.sentiment_service as sentiment_service
        mp.setattr(sentiment_service, 'analyze_texts', mock_analyze_texts)
        
        # Analyze the articles
        enriched_articles = sentiment_service.analyze_articles(articles)
        
        # Check sentiment labels were assigned correctly
        assert enriched_articles[0].sentiment_label == 'positive'
        assert enriched_articles[1].sentiment_label == 'neutral'
        assert enriched_articles[2].sentiment_label == 'negative'
        
        # Get sentiment scores for trend analysis
        sentiment_scores = [a.sentiment_score for a in enriched_articles]
        
        # Build trend DataFrame
        trend_df = build_trend_dataframe(stock_data.data, sentiment_scores)
        
        # Verify trend data has the right structure
        assert 'time' in trend_df.columns
        assert 'text_score' in trend_df.columns
        assert 'price' in trend_df.columns
        assert len(trend_df) == min(len(sentiment_scores), len(stock_data.data))
    
    print("✓ End-to-end workflow functions correctly")
    return True

if __name__ == "__main__":
    # Run the tests directly
    tests_passed = True
    tests_passed &= test_yahoo_finance_data_handling()
    tests_passed &= test_azure_openai_integration()
    tests_passed &= test_dataframe_1d_arrays()
    tests_passed &= test_end_to_end_workflow()
    
    if tests_passed:
        print("\n✅ All tests passed! The modularized code correctly addresses known issues.")
    else:
        print("\n❌ Some tests failed. Further fixes are needed.")
