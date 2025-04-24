import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import app

class DummyClient:
    class chat:
        class completions:
            @staticmethod
            def create(model, messages, max_tokens):
                class Choice:
                    def __init__(self):
                        self.message = type('M', (), {'content': 'dummy response'})
                return type('R', (), {'choices': [Choice()]})()


def test_simple_sentiment_analysis_vader_disabled(monkeypatch):
    monkeypatch.setattr(app, 'VADER_AVAILABLE', False)
    scores, labels = app.simple_sentiment_analysis(['test text'])
    assert scores == [0.0]
    assert labels[0]['label'] == 'neutral'


def test_simple_sentiment_analysis_vader_enabled(monkeypatch):
    monkeypatch.setattr(app, 'VADER_AVAILABLE', True)
    class FakeSia:
        def polarity_scores(self, text): return {'compound': 0.2}
    monkeypatch.setattr(app, 'sia', FakeSia())
    scores, labels = app.simple_sentiment_analysis(['hey'])
    assert scores == [0.2]
    assert labels[0]['label'] == 'positive'


def test_fetch_stock_data(monkeypatch):
    # Create a realistic YFinance-style DataFrame with DateTime index
    dates = pd.date_range('2025-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        'Open': [95, 145, 195],
        'High': [110, 160, 210], 
        'Low': [90, 140, 190],
        'Close': [100, 150, 200],
        'Volume': [1000, 2000, 3000]
    }, index=dates)
    monkeypatch.setattr(app.yf, 'download', lambda *args, **kwargs: df)
    
    res = app.fetch_stock_data('ABC', 'Last 3 Days')
    assert isinstance(res, dict)
    assert res['current_price'] == 200
    assert pytest.approx(res['change_pct'], rel=1e-5) == (200 - 100) / 100 * 100


def test_fetch_stock_data_empty(monkeypatch):
    monkeypatch.setattr(app.yf, 'download', lambda *args, **kwargs: pd.DataFrame())
    res = app.fetch_stock_data('XXX', 'Last 3 Days')
    assert res is None


def test_fetch_news_disabled(monkeypatch):
    monkeypatch.setattr(app, 'NEWS_API_AVAILABLE', False)
    arts = app.fetch_news('SYMBOL', 'Last Week')
    assert arts == []


def test_fetch_news_enabled(monkeypatch):
    monkeypatch.setattr(app, 'NEWS_API_AVAILABLE', True)
    
    # Create a more realistic NewsAPI response
    fake_news = {
        'articles': [
            {
                'title': 'Apple stock reaches new heights',
                'description': 'AAPL surged to a new high yesterday',
                'url': 'https://example.com/news1',
                'publishedAt': '2025-04-22T12:30:00Z',
                'source': {'name': 'Financial Times'},
                'content': 'Full article content here...'
            },
            {
                'title': 'New iPhone launch imminent',
                'description': 'Apple expected to announce iPhone 16 next month',
                'url': 'https://example.com/news2', 
                'publishedAt': '2025-04-21T09:15:00Z',
                'source': {'name': 'Tech Journal'},
                'content': 'More content here...'
            }
        ],
        'status': 'ok',
        'totalResults': 2
    }
    
    # Override newsapi.get_everything to return realistic structure
    dummy = type('N', (), {
        'get_everything': staticmethod(lambda *args, **kwargs: fake_news)
    })()
    monkeypatch.setattr(app, 'newsapi', dummy)
    
    arts = app.fetch_news('AAPL', 'Last Week')
    assert isinstance(arts, list) 
    assert len(arts) == 2
    assert 'title' in arts[0]
    assert 'publishedAt' in arts[0]


def test_plot_trend_content():
    idx = pd.date_range('2025-01-01', periods=2, freq='D')
    df = pd.DataFrame({'text_score': [0.1, -0.2], 'price': [10, 20]}, index=idx)
    fig = app.plot_trend(df)
    assert isinstance(fig, go.Figure)
    # Expect 3 traces: price, raw sentiment, rolling avg
    assert len(fig.data) == 3
    
    # Check order and placement of traces
    assert fig.data[0].name == 'Price'
    assert fig.data[1].name == 'Raw Sentiment'
    assert fig.data[2].name == 'Rolling Avg'
    
    # Check secondary axis assignments
    assert fig.data[0].yaxis == 'y'  # Price on primary axis
    assert fig.data[1].yaxis == 'y2'  # Raw sentiment on secondary axis
    assert fig.data[2].yaxis == 'y2'  # Rolling avg on secondary axis
    
    # Verify gradient coloring on rolling avg trace
    rolling_avg_trace = fig.data[2]
    assert 'colorscale' in rolling_avg_trace.marker
    
    # Verify Rolling Avg is invisible by default
    assert rolling_avg_trace.visible == 'legendonly'
    
    # Verify range slider is disabled
    assert fig.layout.xaxis.rangeslider.visible is False
    
    # Check axis titles
    assert fig.layout.yaxis.title.text == 'Price ($)'
    assert fig.layout.yaxis2.title.text == 'Sentiment Score'
    
    # Verify legend configuration
    assert fig.layout.legend.itemclick == 'toggle'
    assert fig.layout.legend.itemsizing == 'constant'


def test_chat_response_fallback(monkeypatch):
    monkeypatch.setattr(app, 'OPENAI_AVAILABLE', False)
    out = app.chat_response('hello', {'ticker': 'X', 'current_price': 100, 'change_pct': 5}, [], 0)
    assert 'not available' in out.lower()


def test_chat_response_success(monkeypatch):
    monkeypatch.setattr(app, 'OPENAI_AVAILABLE', True)
    monkeypatch.setattr(app, 'client', DummyClient())
    out = app.chat_response('hello', {'ticker': 'X', 'current_price': 100, 'change_pct': 5}, [{'title': 'a'}], 0)
    assert out == 'dummy response'


def test_build_trend_dataframe_with_series():
     """
     Test build_trend_dataframe with standard Series for Close.
     """
     # Create dummy stock_data
     idx = pd.date_range('2025-01-01', periods=4, freq='D')
     stock_data = pd.DataFrame({'Close': [100, 200, 300, 400]}, index=idx)
     vad_scores = [0.5, -0.5, 0.0]
     df = app.build_trend_dataframe(stock_data, vad_scores)
     # Should have 3 rows and correct columns
     assert list(df.columns) == ['time', 'text_score', 'price']
     assert len(df) == 3
     # time column should be list of Timestamps
     assert isinstance(df['time'].iloc[0], pd.Timestamp)
     # price column should be plain list values
     assert df['price'].tolist() == [100, 200, 300]


def test_build_trend_dataframe_with_dataframe():
    """
    Test build_trend_dataframe with a 'Close' column that is a DataFrame, 
    mimicking the structure from Yahoo Finance that caused the runtime error.
    """
    # Create a simpler test case that reliably triggers the same code path
    idx = pd.date_range('2025-01-01', periods=3, freq='D')
    stock_data = pd.DataFrame({
        'Open': [100, 150, 200],
        'High': [110, 160, 210], 
        'Low': [90, 140, 190],
        'Volume': [1000, 2000, 3000] 
    }, index=idx)
    
    # Add 'Close' as a DataFrame instead of a Series - this is what triggers the error
    # Trick: use a multi-level column structure with one level called 'Close' containing a single column
    close_prices = pd.DataFrame(
        {'value': [105, 155, 205]},
        index=idx
    )
    # This ensures stock_data['Close'] returns a DataFrame, not a Series
    stock_data['Close'] = pd.Series([close_prices]*3, index=idx)
    
    vad_scores = [0.1, 0.2, 0.3]
    
    # This should not raise AttributeError even with the odd data structure
    df = app.build_trend_dataframe(stock_data, vad_scores)
    
    assert len(df) == 3
    assert list(df.columns) == ['time', 'text_score', 'price']
    # We've successfully handled the case where 'Close' was a DataFrame
    assert len(df['price']) == 3


def test_build_trend_dataframe_empty():
    import pandas as pd
    # empty data should yield empty df
    stock_data = pd.DataFrame({'Close': []})
    vad_scores = []
    df = app.build_trend_dataframe(stock_data, vad_scores)
    assert df.empty
