import runpy
import os
import pytest


def test_app_starts_without_errors(monkeypatch):
    """
    Run the Streamlit app script in headless mode by executing its main block
    to catch any import-level or runtime errors without launching the UI.
    """
    # Disable external services to avoid real API calls
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('AZURE_OPENAI_ENDPOINT', raising=False)
    monkeypatch.delenv('AZURE_OPENAI_API_VERSION', raising=False)
    monkeypatch.delenv('AZURE_OPENAI_DEPLOYMENT_NAME', raising=False)
    monkeypatch.delenv('NEWS_API_KEY', raising=False)

    script_path = os.path.join(os.path.dirname(__file__), os.pardir, 'app.py')
    # The script should exit cleanly without raising exceptions
    runpy.run_path(script_path, run_name="__main__")
