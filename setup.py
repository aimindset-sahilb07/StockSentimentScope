from setuptools import setup, find_packages

setup(
    name="stock_sentiment_scope",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "yfinance",
        "newsapi-python",
        "nltk",
        "python-dotenv",
        "openai>=1.0.0",
    ],
)
