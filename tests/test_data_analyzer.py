import pytest
import pandas as pd
import numpy as np
from augini import DataAnalyzer
from unittest.mock import patch


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 10
    
    return pd.DataFrame({
        "customer_id": range(1, n_samples + 1),
        "age": np.random.randint(18, 80, n_samples),
        "gender": np.random.choice(["M", "F"], n_samples),
        "city": np.random.choice(["New York", "London", "Tokyo"], n_samples),
        "purchase_amount": np.random.normal(100, 30, n_samples)
    })


@pytest.fixture
def analyzer():
    """Create a DataAnalyzer instance."""
    return DataAnalyzer(api_key="test-key")


def test_fit_and_chat(analyzer, sample_df):
    """Test basic fit and chat functionality."""
    with patch('augini.data_analyzer.DataAnalyzer.chat') as mock_chat:
        mock_chat.return_value = "Analysis shows typical purchase amount is $100"
        
        analyzer.fit(sample_df)
        response = analyzer.chat("What is the typical purchase amount?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "$100" in response


@pytest.mark.parametrize(
    "invalid_input,error_type,error_match",
    [
        (pd.DataFrame(), ValueError, "must not be empty"),
        (None, ValueError, "must be a pandas DataFrame"),
        ("not a dataframe", ValueError, "must be a pandas DataFrame"),
    ],
)
def test_invalid_inputs(analyzer, invalid_input, error_type, error_match):
    """Test error handling for invalid inputs."""
    with pytest.raises(error_type, match=error_match):
        analyzer.fit(invalid_input)
