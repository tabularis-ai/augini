"""Test the DataAnalyzer class."""
import os
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
    
    # Generate test data
    age = np.random.normal(35, 5, size)
    income = 40000 + 500 * age + np.random.normal(0, 5000, size)
    satisfaction = 7 - 0.1 * age + np.random.normal(0, 1, size)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'satisfaction': satisfaction.round(2),
        'category': np.random.choice(['A', 'B', 'C'], size)
    })
    
    # Add some missing values
    df.loc[np.random.rand(size) < 0.1, 'income'] = np.nan
    
    return df


def test_analyzer_initialization(config):
    """Test DataAnalyzer initialization."""
    analyzer = DataAnalyzer(config)
    assert isinstance(analyzer, DataAnalyzer)
    assert analyzer.config == config


def test_missing_values_analysis(sample_df, config):
    """Test analysis of missing values."""
    analyzer = DataAnalyzer(config)
    result = analyzer.analyze_missing_values(sample_df)
    
    assert isinstance(result, dict)
    assert result  # Result should not be empty
    assert all(isinstance(v, (int, float, list, dict)) for v in result.values())


def test_correlation_analysis(sample_df, config):
    """Test analysis of correlations."""
    analyzer = DataAnalyzer(config)
    result = analyzer.analyze_correlations(sample_df)
    
    assert isinstance(result, dict)
    assert result  # Result should not be empty
    assert all(isinstance(v, (list, dict)) for v in result.values())


def test_statistical_analysis(sample_df, config):
    """Test statistical analysis."""
    analyzer = DataAnalyzer(config)
    result = analyzer.analyze_statistics(sample_df)
    
    assert isinstance(result, dict)
    assert result  # Result should not be empty
    assert all(isinstance(v, (int, float, list, dict)) for v in result.values())


def test_complex_analysis(sample_df, config):
    """Test complex analysis combining multiple tools."""
    analyzer = DataAnalyzer(config)
    result = analyzer.analyze(sample_df, "Analyze the relationship between age and income.")
    
    assert isinstance(result, dict)
    assert result.get("success") is not None
    assert result.get("question") == "Analyze the relationship between age and income."
    if result["success"]:
        assert result.get("answer")
    else:
        assert result.get("error")


def test_invalid_question(sample_df, config):
    """Test handling of invalid/empty questions."""
    analyzer = DataAnalyzer(config)
    with pytest.raises(ValueError):
        analyzer.analyze(sample_df, "")


def test_empty_dataframe(config):
    """Test handling of empty DataFrame."""
    analyzer = DataAnalyzer(config)
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        analyzer.analyze(df, "Any analysis")


def test_configuration_update(sample_df, config):
    """Test updating configuration during analysis."""
    analyzer = DataAnalyzer(config)
    
    # Update configuration
    new_config = AuginiConfig(
        llm=LLMConfig(
            provider=LLMProvider.OPENROUTER,
            api_key="test-api-key",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-sonnet",
            temperature=0.5  # Changed temperature
        ),
        agent=AgentConfig(
            max_iterations=5,  # Changed max iterations
            early_stopping=True,
            return_intermediate_steps=True
        ),
        tools=config.tools,
        debug=True
    )
    
    analyzer.update_config(new_config)
    assert analyzer.config.llm.temperature == 0.5
    assert analyzer.config.agent.max_iterations == 5 