"""
Main dashboard UI for StockSentimentScope.
Handles the Streamlit UI layout and interaction.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from stock_sentiment_scope.services.stock_service import fetch_stock_data
from stock_sentiment_scope.services.news_service import fetch_news
from stock_sentiment_scope.services.sentiment_service import (
    analyze_articles, 
    build_trend_dataframe,
    get_sentiment_summary
)
from stock_sentiment_scope.services.chat_service import chat_response
from stock_sentiment_scope.ui.components import (
    plot_trend, 
    sentiment_icon, 
    render_article_card,
    render_metric_cards,
    apply_custom_styles,
    render_footer
)
from stock_sentiment_scope.models.data_models import ChatContext, StockData, Article
from stock_sentiment_scope.utils.logging import get_logger

logger = get_logger(__name__)

def sort_articles(articles: List[Article], sort_by: str) -> List[Article]:
    """
    Sort articles based on specified criteria.
    
    Args:
        articles: List of Article objects
        sort_by: Sort criteria ('Newest First', 'Oldest First', 'Strongest Sentiment')
        
    Returns:
        Sorted list of articles
    """
    if sort_by == "Newest First":
        return sorted(
            articles, 
            key=lambda x: x.published_at if x.published_at else pd.Timestamp.min, 
            reverse=True
        )
    elif sort_by == "Oldest First":
        return sorted(
            articles, 
            key=lambda x: x.published_at if x.published_at else pd.Timestamp.max
        )
    else:  # Strongest Sentiment
        return sorted(
            articles, 
            key=lambda x: abs(x.sentiment_score), 
            reverse=True
        )

def filter_articles_by_sentiment(
    articles: List[Article], 
    sentiment: str
) -> List[Article]:
    """
    Filter articles by sentiment.
    
    Args:
        articles: List of Article objects
        sentiment: Sentiment to filter by ('positive', 'neutral', 'negative')
        
    Returns:
        Filtered list of articles
    """
    return [a for a in articles if a.sentiment_label == sentiment.lower()]

def handle_deep_dive(article: Article, stock: StockData, articles: List[Article], avg_sentiment: float):
    """
    Handle deep dive analysis of an article.
    
    Args:
        article: Article to analyze
        stock: Stock data
        articles: All articles
        avg_sentiment: Average sentiment score
    """
    if 'history' not in st.session_state:
        st.session_state.history = []
        
    user_prompt = f"Summarize the article: '{article.title}'"
    st.session_state.history.append(("user", user_prompt))
    
    with st.spinner("Generating summary..."):
        article_content = article.description or article.content or article.title
        
        if article_content:
            summary_prompt = (
                f"Summarize the following article for a finance-interested audience.\n\n"
                f"Title: {article.title}\n"
                f"Content: {article_content}\n\n"
                f"Please provide your response in the following format:\n"
                f"Key Takeaways:\n- ...\n- ...\n"
                f"Conclusion:\n..."
            )
            
            # Create a ChatContext for the chat service
            chat_context = ChatContext(
                stock=stock,
                articles=articles,
                overall_sentiment=avg_sentiment
            )
            
            resp = chat_response(summary_prompt, chat_context)
            st.session_state.history.append(("assistant", resp))
        else:
            st.session_state.history.append(("assistant", "Sorry, this article does not have enough content to summarize."))
            
    st.rerun()

def display_articles(
    filtered_articles: List[Article],
    sort_by: str,
    stock: StockData,
    all_articles: List[Article],
    avg_sentiment: float
):
    """
    Display articles with pagination.
    
    Args:
        filtered_articles: Articles to display (already filtered)
        sort_by: How to sort the articles
        stock: Stock data
        all_articles: All articles (for context in deep dive)
        avg_sentiment: Average sentiment score
    """
    # Initialize session state for article pagination
    if 'articles_shown' not in st.session_state:
        st.session_state.articles_shown = 5
        
    # Sort articles
    sorted_arts = sort_articles(filtered_articles, sort_by)
    st.markdown(f"#### {len(sorted_arts)} Articles")
    
    # Display articles
    articles_displayed = 0
    max_to_show = st.session_state.articles_shown
    
    for i, article in enumerate(sorted_arts):
        if articles_displayed >= max_to_show:
            break
            
        render_article_card(
            article, 
            articles_displayed,
            on_deep_dive=lambda a=article: handle_deep_dive(a, stock, all_articles, avg_sentiment)
        )
        articles_displayed += 1
    
    # Load more button
    if articles_displayed < len(sorted_arts):
        if st.button("Load more articles"):
            st.session_state.articles_shown += 5
            st.rerun()
    else:
        st.session_state.articles_shown = 5
        
    # Sentiment legend
    st.markdown("""
    <div style='margin-top: 12px; margin-bottom: 8px; font-size: 0.95em;'>
    <b>Sentiment Legend:</b> 🟢 Positive &nbsp;&nbsp; 🟡 Neutral &nbsp;&nbsp; 🔴 Negative
    </div>
    """, unsafe_allow_html=True)
    
def display_chat(
    stock: StockData,
    articles: List[Article],
    avg_sentiment: float
):
    """
    Display the chat interface.
    
    Args:
        stock: Stock data
        articles: All articles
        avg_sentiment: Average sentiment score
    """
    from stock_sentiment_scope.config import OPENAI_AVAILABLE
    
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
                # Create a ChatContext for the chat service
                chat_context = ChatContext(
                    stock=stock,
                    articles=articles,
                    overall_sentiment=avg_sentiment
                )
                
                resp = chat_response(prompt, chat_context)
                
            st.session_state.history.append(("user", prompt))
            st.session_state.history.append(("assistant", resp))
            st.rerun()
    else:
        st.warning("Chat assistant is not available. Please check your Azure OpenAI configuration.")

def run_app():
    """
    Main entry point for the Streamlit app.
    """
    # Set page config
    st.set_page_config(page_title="StockSentimentScope", page_icon="📈", layout="wide")
    
    # Apply custom styles
    apply_custom_styles()
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    
    # App title and subtitle
    st.title("📈 StockSentimentScope")
    st.subheader("Real-time AI‑powered sentiment analysis from latest news articles")
    
    # Sidebar inputs
    ticker = st.sidebar.text_input("Ticker", value="AAPL", key="ticker_input")
    
    # Native Streamlit radio toggle for time frame
    toggle_labels = ["24hr", "3D", "7D", "30D"]
    toggle_map = {
        "24hr": "Last 24 Hours",
        "3D": "Last 3 Days",
        "7D": "Last Week",
        "30D": "Last Month"
    }
    selected_toggle = st.sidebar.radio(
        "Time Frame",
        toggle_labels,
        index=1,
        key="time_frame_toggle",
        horizontal=True
    )
    time_frame = toggle_map[selected_toggle]
    
    if st.sidebar.button("Analyze Sentiment", key="analyze_btn"):
        # Check if we're analyzing a new ticker and clear chat history if so
        if 'last_analyzed_ticker' in st.session_state and st.session_state.last_analyzed_ticker != ticker:
            if 'history' in st.session_state:
                st.session_state.history = []
        
        # Store current ticker for future comparison
        st.session_state.last_analyzed_ticker = ticker
        st.session_state.analysis_done = True
    
    # Main content area
    if st.session_state.analysis_done:
        with st.spinner("Analyzing stock data and news sentiment..."):
            # Fetch stock data
            stock = fetch_stock_data(ticker, time_frame)
            if not stock: 
                st.error("Failed fetching stock data. Please check the ticker symbol and try again.")
                st.stop()
            
            # Fetch news articles
            raw_articles = fetch_news(ticker, time_frame)
            if not raw_articles:
                st.warning("No news articles found for this time period. Try a different time frame.")
            
            # Extract texts for sentiment analysis
            articles = analyze_articles(raw_articles)
            
            # Get texts with content
            texts = [a.description or a.title for a in articles if a.description or a.title]
            
            if texts:
                # Extract sentiment scores for trend analysis
                sentiment_scores = [a.sentiment_score for a in articles]
                
                # Build trend DataFrame
                df = build_trend_dataframe(stock.data, sentiment_scores)
                
                # Get sentiment summary
                sentiment_summary = get_sentiment_summary(articles)
                avg_sentiment = sentiment_summary['avg_sentiment']
                
                # Display metric cards
                render_metric_cards(stock, sentiment_summary)
                
                # Display trend chart
                st.markdown('<div class="graph-section">', unsafe_allow_html=True)
                st.plotly_chart(plot_trend(df.set_index('time')), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Analysis & Insights section
                st.subheader("Analysis & Insights")
                
                # Split-screen layout for Articles and Chat
                col_articles, col_chat = st.columns([3, 2], gap="large")
                
                with col_articles:
                    # Wrap news section in styled div
                    st.markdown('<div class="news-section">', unsafe_allow_html=True)
                    st.subheader("News Articles Analysis")
                    
                    # Article display controls
                    sentiment_options = ["Positive", "Negative", "Neutral"]
                    sort_options = ["Newest First", "Oldest First", "Strongest Sentiment"]
                    
                    # Place dropdowns in a single row
                    col_sentiment, col_sort = st.columns([1, 1])
                    with col_sentiment:
                        selected_sentiment = st.selectbox(
                            "Show Articles With Sentiment", 
                            sentiment_options, 
                            index=0, 
                            key="sentiment_filter"
                        )
                    with col_sort:
                        sort_by = st.selectbox(
                            "Sort Articles By", 
                            sort_options, 
                            key="sort_by"
                        )
                    
                    # Filter articles by selected sentiment
                    filtered_articles = filter_articles_by_sentiment(articles, selected_sentiment)
                    
                    # Display articles
                    display_articles(filtered_articles, sort_by, stock, articles, avg_sentiment)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_chat:
                    # Wrap chat section in styled div
                    st.markdown('<div class="chat-section">', unsafe_allow_html=True)
                    st.subheader("AI Assistant")
                    
                    # Display chat interface
                    display_chat(stock, articles, avg_sentiment)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No text content found in news articles to analyze.")
    else:
        st.info("Enter a ticker and click 'Analyze Sentiment'")
    
    # Add keyboard shortcuts
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
    
    # Render footer
    render_footer()

if __name__ == "__main__":
    run_app()
