"""
Chat service for StockSentimentScope.
Handles generating responses using Azure OpenAI.
"""
from typing import List, Dict, Any, Optional

import streamlit as st

from stock_sentiment_scope.config import (
    get_openai_client, 
    get_azure_deployment_name,
    OPENAI_AVAILABLE
)
from stock_sentiment_scope.models.data_models import StockData, Article, ChatContext
from stock_sentiment_scope.utils.logging import get_logger

logger = get_logger(__name__)

def chat_response(
    prompt: str, 
    context: ChatContext, 
    max_tokens: int = 1024
) -> str:
    """
    Generate a chat response using Azure OpenAI.
    
    Args:
        prompt: User's prompt or question
        context: ChatContext object with stock data, articles, and sentiment
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated response text
    """
    if not OPENAI_AVAILABLE:
        return "Chat assistant is not available. Please check your Azure OpenAI configuration."
    
    client = get_openai_client()
    if not client:
        return "Chat assistant is not available. Azure OpenAI client could not be initialized."
    
    deployment_name = get_azure_deployment_name()
    
    try:
        # Extract context information
        stock = context.stock
        news = context.articles[:5] if context.articles else []  # Limit to 5 articles for context
        overall_sentiment = context.overall_sentiment
        
        # Format messages for ChatCompletion API
        messages = [
            {
                "role": "system", 
                "content": (
                    f"You are a financial sentiment analyst assistant for StockSentimentScope. "
                    f"Provide concise insights about stock {stock.ticker} based on recent news and sentiment data."
                )
            },
            {
                "role": "user", 
                "content": (
                    f"{prompt}\n\n"
                    f"Context:\n"
                    f"- Stock: {stock.ticker} price ${stock.current_price:.2f}, change {stock.change_pct:.2f}%\n"
                    f"- Overall sentiment score: {overall_sentiment:.2f}\n"
                    f"- Recent headlines:\n" + 
                    "\n".join([f"  * {a.title}" for a in news])
                )
            }
        ]
        
        logger.info(f"Calling Azure OpenAI with deployment name: {deployment_name}")
        
        # Call the chat completions API with proper deployment name
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"Error generating chat response: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return "Sorry, I'm having trouble analyzing this right now. Please try again later."
