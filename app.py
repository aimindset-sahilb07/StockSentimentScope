import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Add custom CSS for sentiment styling and animated dots
st.markdown("""
<style>
.sentiment-positive { background-color: rgba(0, 255, 0, 0.08); padding: 10px; border-radius: 5px; margin: 5px 0; }
.sentiment-neutral { background-color: rgba(255, 193, 7, 0.08); padding: 10px; border-radius: 5px; margin: 5px 0; }
.sentiment-negative { background-color: rgba(255, 0, 0, 0.08); padding: 10px; border-radius: 5px; margin: 5px 0; }
.sentiment-dot {
  display: inline-block;
  width: 14px;
  height: 14px;
  margin: 0 2px;
  border-radius: 50%;
  box-shadow: 0 1px 6px rgba(0,0,0,0.07);
  opacity: 0.7;
}
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
    df2 = df.copy()
    df2['sentiment'] = df2['text_score']
    df2['rolling_avg'] = df2['sentiment'].rolling(3, min_periods=1).mean()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # raw sentiment
    fig.add_trace(
        go.Scatter(
            x=df2.index, y=df2['sentiment'],
            mode='markers',
            marker=dict(color=df2['sentiment'].apply(lambda v: 'green' if v>=0 else 'red')),
            name='Raw Sentiment',
            hovertemplate='Date: %{x}<br>Sentiment: %{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    # rolling average
    fig.add_trace(
        go.Scatter(
            x=df2.index, y=df2['rolling_avg'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Rolling Avg',
            hovertemplate='Date: %{x}<br>Rolling Avg: %{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    # price on secondary axis
    if 'price' in df2.columns:
        fig.add_trace(
            go.Scatter(
                x=df2.index, y=df2['price'],
                mode='lines',
                line=dict(color='orange'),
                name='Price'
            ),
            secondary_y=True
        )
    # layout
    fig.update_layout(
        title='Sentiment Trend',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_rangeslider_visible=True
    )
    fig.update_yaxes(title_text='Price', secondary_y=True)
    return fig

def build_trend_dataframe(stock_data, vad_scores):
    """
    Build time-series DataFrame for sentiment trend chart
    """
    score_len = min(len(vad_scores), len(stock_data))
    # Handle both Series and DataFrame cases for Close prices
    close_data = stock_data['Close']
    if isinstance(close_data, pd.DataFrame):
        # If 'Close' is a DataFrame (happens with some Yahoo API responses)
        price_values = close_data.iloc[:score_len].values.flatten().tolist()
    else:
        # If 'Close' is a Series (standard case)
        price_values = close_data.iloc[:score_len].tolist()
        
    # Create DataFrame with guaranteed 1D columns
    return pd.DataFrame({
        'time': list(stock_data.index[:score_len]),
        'text_score': vad_scores[:score_len],
        'price': price_values
    })

# Chat
def chat_response(msg, stock, news, overall, max_tokens=1024):
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
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating chat response: {str(e)}")
        return "Sorry, I'm having trouble analyzing this right now. Please try again later."

# UI
st.title("📈 StockSentimentScope")
st.subheader("Real-time AI‑powered sentiment analysis from latest news articles")
ticker = st.sidebar.text_input("Ticker", value="AAPL", key="ticker_input")
time_frame = st.sidebar.selectbox(
    "Time Frame",
    ['Last 24 Hours','Last 3 Days','Last Week','Last Month'],
    index=1,
    key="time_frame"
)

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
            
            # Build trend DataFrame via helper to ensure 1D lists
            df = build_trend_dataframe(stock['data'], vad_scores)
            
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
            
            st.plotly_chart(plot_trend(df.set_index('time')), use_container_width=True)
            
            # --- Split-screen layout for Articles and Chat ---
            st.subheader("Analysis & Insights")
            # Give chat more space to prevent overflow
            col_articles, col_chat = st.columns([3, 2], gap="large")

            with col_articles:
                st.subheader("News Articles Analysis")
                # --- Article Display Controls ---
                sentiment_options = ["Positive", "Negative", "Neutral"]
                sentiment_map = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}
                sort_options = ["Newest First", "Oldest First", "Strongest Sentiment"]
                # Place dropdowns in a single row
                col_sentiment, col_sort = st.columns([1, 1])
                with col_sentiment:
                    selected_sentiment = st.selectbox("Show Articles With Sentiment", sentiment_options, index=0, key="sentiment_filter")
                with col_sort:
                    sort_by = st.selectbox("Sort Articles By", sort_options, key="sort_by")

                # --- Annotate articles with sentiment and date before filtering ---
                for i, a in enumerate(articles):
                    a['sentiment_label'] = fin_scores[i]['label'] if i < len(fin_scores) else 'neutral'
                    a['sentiment_score'] = fin_scores[i]['score'] if i < len(fin_scores) else 0.0
                    date_str = a.get('publishedAt', '')
                    try:
                        a['parsed_date'] = pd.to_datetime(date_str)
                    except Exception:
                        a['parsed_date'] = pd.NaT

                # Filter articles by selected sentiment
                filtered_articles = [a for a in articles if a.get('sentiment_label') == sentiment_map[selected_sentiment]]

                def sort_articles(arts):
                    if sort_by == "Newest First":
                        return sorted(arts, key=lambda x: x['parsed_date'] if not pd.isna(x['parsed_date']) else pd.Timestamp.min, reverse=True)
                    elif sort_by == "Oldest First":
                        return sorted(arts, key=lambda x: x['parsed_date'] if not pd.isna(x['parsed_date']) else pd.Timestamp.max)
                    else:  # Strongest Sentiment
                        return sorted(arts, key=lambda x: abs(x['sentiment_score']), reverse=True)

                # --- Pagination/Load More ---
                if 'articles_shown' not in st.session_state:
                    st.session_state.articles_shown = 5

                articles_displayed = 0
                max_to_show = st.session_state.articles_shown
                stop_display = False

                def sentiment_icon(label):
                    if label == 'positive': return '🟢'
                    if label == 'negative': return '🔴'
                    return '🟡'

                sorted_arts = sort_articles(filtered_articles)
                st.markdown(f"#### {selected_sentiment} Articles ({len(sorted_arts)})")
                for i, a in enumerate(sorted_arts):
                    if articles_displayed >= max_to_show:
                        stop_display = True
                    with st.container():
                        title_link = f"[{a['title']}]({a['url']})"
                        st.markdown(f"**{title_link}** {sentiment_icon(a['sentiment_label'])}", unsafe_allow_html=True)
                        meta = f"{a.get('source', {}).get('name', 'Unknown Source')} | "
                        meta += a['parsed_date'].strftime('%b %d, %Y') if not pd.isna(a['parsed_date']) else 'Unknown date'
                        st.caption(meta)
                        if a.get('description'):
                            st.markdown(a['description'])
                        with st.expander("Show full article"):
                            st.write(a.get('content', 'No content available.'))
                        summarize_key = f"summarize_{articles_displayed}"
                        if st.button("Summarize", key=summarize_key):
                            if 'history' not in st.session_state:
                                st.session_state.history = []
                            user_prompt = f"Summarize the article: '{a['title']}'"
                            st.session_state.history.append(("user", user_prompt))
                            with st.spinner("Generating summary..."):
                                article_content = a.get('description') or a.get('content') or a.get('title')
                                if article_content:
                                    summary_prompt = f"Summarize the following article for a finance-interested audience.\n\nTitle: {a['title']}\nContent: {article_content}\n\nPlease provide your response in the following format:\nKey Takeaways:\n- ...\n- ...\nConclusion:\n..."
                                    resp = chat_response(summary_prompt, stock, articles, avg_sentiment)
                                    st.session_state.history.append(("assistant", resp))
                                else:
                                    st.session_state.history.append(("assistant", "Sorry, this article does not have enough content to summarize."))
                            st.rerun()
                    articles_displayed += 1

                # --- Load More Button ---
                if articles_displayed < len(sorted_arts):
                    if st.button("Load more articles"):
                        st.session_state.articles_shown += 5
                        st.rerun()
                else:
                    st.session_state.articles_shown = 5

                # --- Sentiment Legend ---
                st.markdown("""
                <div style='margin-top: 12px; margin-bottom: 8px; font-size: 0.95em;'>
                <b>Sentiment Legend:</b> 🟢 Positive &nbsp;&nbsp; 🟡 Neutral &nbsp;&nbsp; 🔴 Negative
                </div>
                """, unsafe_allow_html=True)

            # --- Chat Interface in right column ---
            with col_chat:
                st.subheader("Chat Assistant")
                # Add custom CSS to ensure chat wraps and fits in column
                st.markdown("""
                <style>
                .element-container .stChatMessageContent, .element-container .stMarkdown {
                    overflow-wrap: break-word !important;
                    word-break: break-word !important;
                    max-width: 100% !important;
                }
                .stChatInputContainer { max-width: 100% !important; }
                </style>
                """, unsafe_allow_html=True)
                if OPENAI_AVAILABLE:
                    if 'history' not in st.session_state:
                        st.session_state.history = []

                    # Display messages
                    for role, msg in st.session_state.history:
                        if role == "user":
                            st.chat_message("user").write(msg)
                        else:
                            st.chat_message("assistant").write(msg)

                    # Input at bottom
                    prompt = st.chat_input("Ask about this analysis")
                    if prompt:
                        with st.spinner("Generating response..."):
                            resp = chat_response(prompt, stock, articles, avg_sentiment)
                        st.session_state.history.append(("user", prompt))
                        st.session_state.history.append(("assistant", resp))
                        st.rerun()
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
