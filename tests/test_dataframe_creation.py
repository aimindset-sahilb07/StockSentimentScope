import pandas as pd
import pytest


def generate_stock_data(n):
    """
    Generates dummy stock data with a date index and Close prices.
    """
    idx = pd.date_range('2025-01-01', periods=n, freq='D')
    data = pd.DataFrame({'Close': [i * 10 for i in range(n)]}, index=idx)
    return data


def create_df(vad_scores, stock_data):
    """
    Mimics app's DataFrame construction logic for sentiment trend.
    """
    score_len = min(len(vad_scores), len(stock_data))
    return pd.DataFrame({
        'time': list(stock_data.index[:score_len]),
        'text_score': vad_scores[:score_len],
        'price': stock_data['Close'].iloc[:score_len].tolist()
    })


def test_create_df_success():
    stock_data = generate_stock_data(5)
    vad_scores = [0.1, -0.2, 0.0, 0.5, -0.1]
    df = create_df(vad_scores, stock_data)
    assert df.shape == (5, 3)
    assert list(df.columns) == ['time', 'text_score', 'price']
    assert df['price'].tolist() == [0, 10, 20, 30, 40]


def test_create_df_vad_longer():
    stock_data = generate_stock_data(3)
    vad_scores = [0.1, 0.2, 0.3, 0.4]
    df = create_df(vad_scores, stock_data)
    # Should trim to stock_data length
    assert df.shape == (3, 3)
    assert df['price'].tolist() == [0, 10, 20]


def test_create_df_empty():
    stock_data = generate_stock_data(0)
    vad_scores = []
    df = create_df(vad_scores, stock_data)
    assert df.empty
