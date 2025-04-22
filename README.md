# StockSentimentScope

A Streamlit app that provides sentiment analysis for stocks using VADER and FinBERT, plus an LLM-powered chat assistant.

## Features
- Input stock ticker and time frame
- Fetch historical stock prices with retry logic
- Pull related news via NewsAPI
- Two sentiment engines: VADER & FinBERT
- Interactive Plotly sentiment trend chart
- Latest headlines with per-article sentiment highlighting
- Enhanced metrics display with overall and dominant sentiment
- Chat assistant powered by Azure OpenAI GPT-4o
- Robust error handling throughout the application

## Technical Details
- Frontend: Streamlit with custom styling
- Data Visualization: Plotly interactive charts
- Data Processing: Python, Pandas, NumPy
- Sentiment Analysis: Hugging Face Transformers (FinBERT), NLTK (VADER)
- Market Data: Yahoo Finance API, News API 
- LLM: Azure OpenAI Service (GPT-4o deployment)

## Setup
1.  Fill in your API keys in the `.env` file:
   - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
   - `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
   - `NEWS_API_KEY`: Your News API key
2. Ensure you have a GPT-4o deployment named "stocksentiment-gpt-4o" in your Azure OpenAI resource
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app.py`

## Error Handling
The application includes comprehensive error handling for:
- Failed API requests to Yahoo Finance, News API, and Azure OpenAI
- FinBERT model loading and inference failures (with VADER fallback)
- Missing or invalid data scenarios
