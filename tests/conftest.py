"""Test configuration and fixtures."""

import pytest
from augini import DataAnalyzer, AuginiConfig, LLMConfig, AgentConfig, LLMProvider, ToolConfig
import pandas as pd
import numpy as np


@pytest.fixture
def config():
    """Create a test configuration."""
    return AuginiConfig(
        llm=LLMConfig(
            provider=LLMProvider.OPENROUTER,
            api_key="test-api-key",  # Mock API key for testing
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-sonnet",
            temperature=0.2
        ),
        agent=AgentConfig(
            max_iterations=3,
            early_stopping=True,
            return_intermediate_steps=True
        ),
        tools={
            "missing_values_analyzer": ToolConfig(
                name="missing_values_analyzer",
                description="Analyzes missing values in the DataFrame",
                enabled=True
            ),
            "correlation_analyzer": ToolConfig(
                name="correlation_analyzer",
                description="Analyzes correlations between columns",
                enabled=True
            ),
            "statistical_summary": ToolConfig(
                name="statistical_summary",
                description="Provides statistical summaries of the data",
                enabled=True
            ),
            "visualization_tool": ToolConfig(
                name="visualization_tool",
                description="Generates visualizations from DataFrame data",
                enabled=True
            )
        },
        debug=True
    )


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    size = 100
    
    # Generate correlated data
    age = np.random.normal(35, 10, size)
    income = 20000 + 1000 * age + np.random.normal(0, 10000, size)
    satisfaction = 7 - 0.1 * age + np.random.normal(0, 2, size)
    
    df = pd.DataFrame({
        'age': age.round(),
        'income': income.round(),
        'satisfaction': satisfaction.round(2),
        'category': np.random.choice(['A', 'B', 'C'], size)
    })
    
    # Add missing values
    df.loc[np.random.rand(size) < 0.1, 'age'] = np.nan
    df.loc[np.random.rand(size) < 0.1, 'income'] = np.nan
    
    return df


@pytest.fixture
def analyzer(config):
    """Create a DataAnalyzer instance."""
    return DataAnalyzer(config)
