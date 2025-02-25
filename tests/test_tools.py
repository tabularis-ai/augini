"""Test data analysis tools."""

import pytest
import pandas as pd
import numpy as np
from augini.tools.missing_values import MissingValuesAnalyzer
from augini.tools.correlation import CorrelationAnalyzer
from augini.tools.statistics import StatisticalSummary
from augini.config.base import ToolConfig


@pytest.fixture
def tool_config():
    """Create a test tool configuration."""
    return ToolConfig(
        name="test_tool",
        description="Test tool",
        enabled=True,
        required_packages=[],
        timeout_seconds=30
    )


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    size = 100
    
    # Generate test data
    age = np.random.normal(35, 5, size)
    income = 40000 + 500 * age + np.random.normal(0, 5000, size)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'category': np.random.choice(['A', 'B', 'C'], size)
    })
    
    # Add some missing values
    df.loc[np.random.rand(size) < 0.1, 'income'] = np.nan
    
    return df


def test_missing_values_analyzer(sample_df, tool_config):
    """Test MissingValuesAnalyzer functionality."""
    analyzer = MissingValuesAnalyzer(config=tool_config)
    result = analyzer.analyze(sample_df)
    
    assert isinstance(result, dict)
    assert result  # Result should not be empty
    assert all(isinstance(v, (int, float, list, dict)) for v in result.values())


def test_correlation_analyzer(sample_df, tool_config):
    """Test CorrelationAnalyzer functionality."""
    analyzer = CorrelationAnalyzer(config=tool_config)
    result = analyzer.analyze(sample_df)
    
    assert isinstance(result, dict)
    assert result  # Result should not be empty
    assert all(isinstance(v, (list, dict)) for v in result.values())


def test_statistical_summary(sample_df, tool_config):
    """Test StatisticalSummary functionality."""
    analyzer = StatisticalSummary(config=tool_config)
    result = analyzer.analyze(sample_df)
    
    assert isinstance(result, dict)
    assert result  # Result should not be empty
    assert all(isinstance(v, (int, float, list, dict)) for v in result.values())


def test_disabled_tool(sample_df):
    """Test that disabled tools raise an error."""
    config = ToolConfig(
        name="disabled_tool",
        description="Disabled tool",
        enabled=False
    )
    
    with pytest.raises(ValueError, match="Tool disabled_tool is disabled"):
        MissingValuesAnalyzer(config=config)


def test_empty_dataframe(tool_config):
    """Test tools with empty DataFrame."""
    df = pd.DataFrame()
    
    for Tool in [MissingValuesAnalyzer, CorrelationAnalyzer, StatisticalSummary]:
        analyzer = Tool(config=tool_config)
        with pytest.raises(ValueError, match="Empty DataFrame provided"):
            analyzer.analyze(df) 