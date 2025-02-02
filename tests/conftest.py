import pytest
import pandas as pd
from augini import DataAnalyzer, DataEngineer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "Department": ["Sales", "Engineering", "Marketing"],
            "Revenue": [100000, 50000, 75000],
            "Team_Size": [10, 25, 15],
            "Growth_Rate": [0.15, 0.25, 0.10],
        }
    )


@pytest.fixture
def analyzer():
    """Create a DataAnalyzer instance."""
    return DataAnalyzer(api_key="test-key")


@pytest.fixture
def engineer():
    """Create a DataEngineer instance."""
    return DataEngineer(api_key="test-key")
