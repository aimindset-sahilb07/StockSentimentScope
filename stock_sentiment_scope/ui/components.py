"""
UI components for StockSentimentScope.
Contains reusable UI components for the dashboard.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

from stock_sentiment_scope.utils.logging import get_logger
from stock_sentiment_scope.models.data_models import Article

logger = get_logger(__name__)

def plot_trend(df):
    """
    Create sentiment trend chart with price overlay.
    
    Args:
        df: DataFrame with time index, sentiment, and price columns
        
    Returns:
        Plotly figure object
    """
    df2 = df.copy()
    df2['sentiment'] = df2['text_score']
    df2['rolling_avg'] = df2['sentiment'].rolling(3, min_periods=1).mean()
    
    # Make a subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # price on primary axis
    if 'price' in df2.columns:
        fig.add_trace(
            go.Scatter(
                x=df2.index, y=df2['price'],
                mode='lines',
                line=dict(color='rgba(255, 165, 0, 0.8)', width=2),  # More transparent orange
                name='Price'
            ),
            secondary_y=False
        )
    
    # raw sentiment on secondary axis
    fig.add_trace(
        go.Scatter(
            x=df2.index, y=df2['sentiment'],
            mode='markers',
            marker=dict(
                color=df2['sentiment'].apply(lambda v: 'green' if v>=0 else 'red'),
                size=8,
                opacity=0.7
            ),
            name='Raw Sentiment',
            hovertemplate='Date: %{x}<br>Sentiment: %{y:.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # rolling average with gradient line - using simpler colorscale approach
    # Create a list of colors based on sentiment values
    colors = ['red' if val < -0.05 else 'yellow' if val < 0.05 else 'green' for val in df2['rolling_avg']]
    
    fig.add_trace(
        go.Scatter(
            x=df2.index, y=df2['rolling_avg'],
            mode='lines',
            line=dict(
                width=2,
                color='rgba(65, 105, 225, 0.6)', # Lighter royal blue with transparency
                dash='dot',  # Make the line dotted
            ),
            marker=dict(
                color=df2['rolling_avg'],
                colorscale=[
                    [0, 'red'],
                    [0.5, 'yellow'],
                    [1.0, 'green']
                ],
                colorbar=dict(title="Sentiment")
            ),
            name='Rolling Avg',
            hovertemplate='Date: %{x}<br>Rolling Avg: %{y:.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # layout
    fig.update_layout(
        title='Sentiment Trend',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='right', 
            x=1,
            # Add clickmode to show that legends are toggles
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            # Add visual indicators to show legend items are clickable
            itemsizing="constant"
        ),
        xaxis_rangeslider_visible=False,  # Removed as requested
        plot_bgcolor='rgba(240,240,240,0.2)',  # Lighter background
        xaxis=dict(
            showgrid=False,  # Remove x grid
            zeroline=True,   # Keep zero line
            zerolinecolor='rgba(0,0,0,0.2)'  # Subtle zero line
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',  # Very subtle grid
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)'
        )
    )
    
    fig.update_yaxes(title_text='Sentiment Score', secondary_y=True)
    fig.update_yaxes(title_text='Price ($)', secondary_y=False)
    
    # Remove horizontal grid lines for secondary y-axis (sentiment)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    
    # Set Rolling Avg trace to be invisible by default
    fig.data[2].visible = 'legendonly'
    
    return fig

def sentiment_icon(label):
    """
    Return emoji icon for sentiment label.
    
    Args:
        label: Sentiment label (positive, neutral, negative)
        
    Returns:
        Emoji string
    """
    if label == 'positive': return '🟢'
    if label == 'negative': return '🔴'
    return '🟡'

def render_article_card(article: Article, index: int, on_deep_dive=None):
    """
    Render a news article card.
    
    Args:
        article: Article object
        index: Article index (for button key)
        on_deep_dive: Callback function for deep dive button
        
    Returns:
        None (renders directly to Streamlit)
    """
    with st.container():
        title_link = f"[{article.title}]({article.url})"
        st.markdown(f"**{title_link}**", unsafe_allow_html=True)
        
        meta = f"{article.source_name or 'Unknown Source'} | "
        meta += article.published_at.strftime('%b %d, %Y') if article.published_at else 'Unknown date'
        st.caption(meta)
        
        if article.description:
            st.markdown(article.description)
        
        summarize_key = f"summarize_{index}"
        if st.button("Deep Dive ->", key=summarize_key) and on_deep_dive:
            on_deep_dive(article)

def render_metric_cards(stock_data, sentiment_summary):
    """
    Render metric cards for stock price and sentiment.
    
    Args:
        stock_data: StockData object
        sentiment_summary: Sentiment summary dict
        
    Returns:
        None (renders directly to Streamlit)
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price", 
            f"${stock_data.current_price:.2f}", 
            f"{stock_data.change_pct:.2f}%"
        )
    
    with col2:
        avg_sentiment = sentiment_summary['avg_sentiment']
        sentiment_label = sentiment_summary['sentiment_label'].capitalize()
        st.metric(
            "Overall Sentiment", 
            f"{sentiment_label}", 
            f"{avg_sentiment:.2f}"
        )
    
    with col3:
        dominant = sentiment_summary['dominant_sentiment']
        count = sentiment_summary['dominant_count']
        total = sentiment_summary['total_count']
        st.metric(
            "Dominant Sentiment", 
            f"{dominant.capitalize()}", 
            f"{count}/{total}"
        )

def apply_custom_styles():
    """
    Apply custom styles to the Streamlit app.
    """
    st.markdown("""
    <style>
      .stApp {
        margin-bottom: 70px;
      }
      
      /* Make sidebar narrower */
      [data-testid="stSidebar"] {
        width: 14rem !important;
      }
      
      /* Adjust sidebar input controls width */
      [data-testid="stSidebar"] section[data-testid="stVerticalBlock"] {
        width: 14rem;
        padding-right: 1rem;
      }
      
      /* Section styling */
      .graph-section {
        background-color: rgba(173, 216, 230, 0.2);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
      }
      
      .news-section {
        background-color: rgba(221, 160, 221, 0.2);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
      }
      
      .chat-section {
        background-color: rgba(144, 238, 144, 0.2);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
      }
      
      /* Chat message styling */
      .element-container .stChatMessageContent, .element-container .stMarkdown {
        overflow-wrap: break-word !important;
        word-break: break-word !important;
        max-width: 100% !important;
      }
      .stChatInputContainer { max-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

def render_footer():
    """
    Render page footer with disclaimer.
    """
    st.markdown("---")
    
    footer_container = st.container()
    
    with footer_container:
        st.markdown(
            """
            <div style="background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); 
                        padding: 10px 0 8px 0; 
                        text-align: center;
                        margin-top: 20px;
                        box-shadow: 0 -1px 6px rgba(0,0,0,0.1);">
                <p style="font-style: italic; margin: 0;">
                    <em>Disclaimer: I'm just a dashboard, not your financial advisor. This information is provided for informational purposes only and should not be considered investment advice. — always do your own research!</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
