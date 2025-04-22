import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta, timezone
from newsapi import NewsApiClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from dotenv import load_dotenv
import time
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="StockSentimentScope", page_icon="📈", layout="wide")

# Initialize session state flag
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Load env
load_dotenv()
# Azure OpenAI client
try:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    )
    # Define model deployment name
    AZURE_DEPLOYMENT_NAME =os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "stocksentiment-gpt-4o")
    OPENAI_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"OpenAI client initialization failed: {str(e)}")
    OPENAI_AVAILABLE = False

# Add custom CSS for sentiment styling
st.markdown("""
<style>
.sentiment-positive { background-color: rgba(0, 255, 0, 0.2); padding: 10px; border-radius: 5px; margin: 5px 0; }
.sentiment-neutral { background-color: rgba(255, 255, 0, 0.2); padding: 10px; border-radius: 5px; margin: 5px 0; }
.sentiment-negative { background-color: rgba(255, 0, 0, 0.2); padding: 10px; border-radius: 5px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# NewsAPI client
try:
    newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
    NEWS_API_AVAILABLE = True
except Exception as e:
    logger.warning(f"NewsAPI client initialization failed: {str(e)}")
    NEWS_API_AVAILABLE = False

# NLTK
try:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception as e:
    logger.warning(f"VADER initialization failed: {str(e)}")
    VADER_AVAILABLE = False

# Global flag to track if FinBERT is available - set to False as default since we're not using it
FINBERT_AVAILABLE = False
finbert = None

# We're not using FinBERT in this version to avoid PyTorch dependency issues
# Define a simpler sentiment analysis function instead
def simple_sentiment_analysis(texts):
    """Simplified sentiment analysis using only VADER"""
    if not VADER_AVAILABLE:
        st.warning("VADER sentiment analysis is not available.")
        return [0.0 for _ in texts], [{'label': 'neutral', 'score': 0.0} for _ in texts]
    
    # Use VADER for sentiment scores
    scores = [sia.polarity_scores(t)['compound'] for t in texts]
    
    # Convert to label format
    labels = [{'label': 'positive' if s > 0.05 else 'negative' if s < -0.05 else 'neutral', 
               'score': abs(s)} for s in scores]
    
    return scores, labels

# Fetch stock data
def fetch_stock_data(ticker, tf):
    end = datetime.now()
    if tf=='Last 24 Hours': start = end - timedelta(days=1); interval='1h'
    elif tf=='Last 3 Days': start = end - timedelta(days=3); interval='1h'
    elif tf=='Last Week': start = end - timedelta(days=7); interval='1d'
    else: start = end - timedelta(days=30); interval='1d'
    df=None; retries=3
    for i in range(retries):
        try:
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
            if not df.empty: break
        except Exception as e:
            logger.warning(f"Attempt {i+1}/{retries} to fetch stock data failed: {str(e)}")
            time.sleep(2)
    if df is None or df.empty: return None
    # Use .item() to extract scalar floats from the Series without float() call
    cur = df['Close'].iloc[-1].item(); prev = df['Close'].iloc[0].item()
    return {'data':df, 'current_price':cur, 'change_pct':((cur-prev)/prev*100)}

# Fetch news
def fetch_news(ticker, tf):
    if not NEWS_API_AVAILABLE:
        st.warning("News API is not available. Unable to fetch news articles.")
        return []
        
    # Use timezone-aware datetime object (fix deprecation warning)
    to = datetime.now(timezone.utc)
    if tf=='Last 24 Hours': frm = to - timedelta(days=1)
    elif tf=='Last 3 Days': frm = to - timedelta(days=3)
    elif tf=='Last Week': frm = to - timedelta(days=7)
    else: frm = to - timedelta(days=30)
    
    try:
        # Format dates correctly as YYYY-MM-DD
        from_date = frm.strftime('%Y-%m-%d')
        to_date = to.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching news for {ticker} from {from_date} to {to_date}")
        res = newsapi.get_everything(
            q=ticker, 
            from_param=from_date, 
            to=to_date, 
            language='en', 
            sort_by='relevancy', 
            page_size=20
        )
        return res.get('articles', [])
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        logger.error(f"NewsAPI error: {str(e)}")
        return []

# Chart
def plot_trend(df):
    df2 = df.copy(); df2['sentiment'] = df2['text_score']
    return px.line(df2, x=df2.index, y='sentiment', title='Sentiment Trend')

# Chat
def chat_response(msg, stock, news, overall):
    if not OPENAI_AVAILABLE:
        return "Chat assistant is not available. Please check your Azure OpenAI configuration."
        
    try:
        # Format messages for ChatCompletion API
        messages = [
            {"role": "system", "content": f"You are a financial sentiment analyst assistant for StockSentimentScope. Provide concise insights about stock {stock['ticker']} based on recent news and sentiment data."},
            {"role": "user", "content": f"{msg}\n\nContext:\n- Stock: {stock['ticker']} price ${stock['current_price']:.2f}, change {stock['change_pct']:.2f}%\n- Overall sentiment score: {overall:.2f}\n- Recent headlines:\n" + "\n".join([f"  * {a['title']}" for a in news[:5]])}
        ]
        
        # Call the chat completions API with proper deployment name
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=150
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating chat response: {str(e)}")
        return "Sorry, I'm having trouble analyzing this right now. Please try again later."

# UI
st.title("StockSentimentScope")
ticker = st.sidebar.text_input("Ticker", value="AAPL", key="ticker_input")
time_frame = st.sidebar.selectbox("Time Frame", ['Last 24 Hours','Last 3 Days','Last Week','Last Month'], key="time_frame")

if st.sidebar.button("Analyze Sentiment", key="analyze_btn"):
    st.session_state.analysis_done = True

if st.session_state.analysis_done:
    with st.spinner("Analyzing stock data and news sentiment..."):
        stock = fetch_stock_data(ticker, time_frame)
        if not stock: 
            st.error("Failed fetching stock data. Please check the ticker symbol and try again.")
            st.stop()
            
        # Add ticker to stock info for reference in chat
        stock['ticker'] = ticker
        
        articles = fetch_news(ticker, time_frame)
        if not articles:
            st.warning("No news articles found for this time period. Try a different time frame.")
        
        texts = [a['description'] or a['title'] for a in articles if a.get('description') or a.get('title')]
        
        if texts:
            # Use the simple sentiment analysis function
            vad_scores, fin_scores = simple_sentiment_analysis(texts)
            
            # Ensure we don't exceed the length of available data
            score_len = min(len(vad_scores), len(stock['data']))
            df = pd.DataFrame({'time':stock['data'].index[:score_len], 'text_score':vad_scores[:score_len]})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${stock['current_price']:.2f}", f"{stock['change_pct']:.2f}%")
            with col2:
                avg_sentiment = np.mean(vad_scores)
                sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
                st.metric("Overall Sentiment", f"{sentiment_label}", f"{avg_sentiment:.2f}")
            with col3:
                sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
                for item in fin_scores:
                    sentiment_counts[item['label']] += 1
                dominant = max(sentiment_counts, key=sentiment_counts.get)
                st.metric("Dominant Sentiment", f"{dominant.capitalize()}", f"{sentiment_counts[dominant]}/{len(fin_scores)}")
            
            fig = plot_trend(df.set_index('time'))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("News Articles Analysis")
            for i, a in enumerate(articles[:10]):  # Limit to first 10 articles for display
                if i < len(vad_scores) and i < len(fin_scores):
                    col = 'sentiment-positive' if vad_scores[i] > 0.05 else 'sentiment-negative' if vad_scores[i] < -0.05 else 'sentiment-neutral'
                    st.markdown(f"<div class='{col}'><a href='{a['url']}' target='_blank'>{a['title']}</a> ({fin_scores[i]['label']} {fin_scores[i]['score']:.2f})</div>", unsafe_allow_html=True)
            
            # Chat
            if OPENAI_AVAILABLE:
                st.subheader("Chat Assistant")
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                with st.form("chat_form", clear_on_submit=False):
                    user_msg = st.text_input("Ask about this analysis:", key="chat_input")
                    send = st.form_submit_button("Send")
                    if send and user_msg:
                        with st.spinner("Generating response..."):
                            resp = chat_response(user_msg, stock, articles, avg_sentiment)
                            st.session_state.history.append((user_msg, resp))
                
                for u, r in st.session_state.history:
                    st.markdown(f"**You:** {u}")
                    st.markdown(f"**Bot:** {r}")
            else:
                st.warning("Chat assistant is not available. Please check your Azure OpenAI configuration.")
        else:
            st.warning("No text content found in news articles to analyze.")
else:
    st.info("Enter a ticker and click 'Analyze Sentiment'")

st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    const btns = document.querySelectorAll('button');
    btns.forEach(b => {
      if (b.innerText.trim() === 'Send') { b.click(); }
    });
  } else if (e.key === 'Enter' && e.shiftKey) {
    const btns = document.querySelectorAll('button');
    btns.forEach(b => {
      if (b.innerText.trim() === 'Analyze Sentiment') { b.click(); }
    });
  }
});
</script>
""", unsafe_allow_html=True)
