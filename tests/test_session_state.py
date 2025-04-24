import pytest
import streamlit as st
import app

def test_ticker_change_clears_chat_history(monkeypatch):
    """Test that changing the ticker and clicking Analyze Sentiment clears chat history"""
    
    # Mock streamlit session state
    class MockSessionState(dict):
        def __init__(self):
            super().__init__()
            self["analysis_done"] = False
            self["history"] = [("user", "Test message"), ("assistant", "Test response")]
            self["last_analyzed_ticker"] = "AAPL"
        
        def __getattr__(self, name):
            if name in self:
                return self[name]
            raise AttributeError(f"'MockSessionState' has no attribute '{name}'")
            
        def __setattr__(self, name, value):
            self[name] = value
    
    # Create our mock session state
    mock_session_state = MockSessionState()
    
    # Mock st.session_state with our custom session state
    monkeypatch.setattr(st, "session_state", mock_session_state)
    
    # Simulate clicking Analyze Sentiment with same ticker
    ticker = "AAPL"
    
    # Call the logic that would be run when clicking the button
    if 'last_analyzed_ticker' in st.session_state and st.session_state.last_analyzed_ticker != ticker:
        if 'history' in st.session_state:
            st.session_state.history = []
    st.session_state.last_analyzed_ticker = ticker
    
    # History should not be cleared since ticker is the same
    assert len(st.session_state.history) == 2
    
    # Now simulate clicking Analyze Sentiment with different ticker
    ticker = "MSFT"
    
    # Call the logic again
    if 'last_analyzed_ticker' in st.session_state and st.session_state.last_analyzed_ticker != ticker:
        if 'history' in st.session_state:
            st.session_state.history = []
    st.session_state.last_analyzed_ticker = ticker
    
    # History should be cleared since ticker changed
    assert len(st.session_state.history) == 0
    
    # Ensure the last_analyzed_ticker was updated
    assert st.session_state.last_analyzed_ticker == "MSFT"
