"""Test the DataChat class."""
import os
import pytest
from augini import DataChat, DataAnalyzer, AuginiConfig, LLMConfig, AgentConfig, LLMProvider, ToolConfig, ChatSession, ChatMessage
import pandas as pd
import numpy as np
import re


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


def test_chat_message():
    """Test ChatMessage initialization and conversion."""
    message = ChatMessage("user", "Hello")
    
    assert message.role == "user"
    assert message.content == "Hello"
    assert message.timestamp is not None
    assert message.id is not None
    
    # Test to_dict and from_dict
    message_dict = message.to_dict()
    assert message_dict["role"] == "user"
    assert message_dict["content"] == "Hello"
    
    new_message = ChatMessage.from_dict(message_dict)
    assert new_message.role == message.role
    assert new_message.content == message.content


def test_chat_session():
    """Test ChatSession functionality."""
    session = ChatSession()
    
    assert session.session_id is not None
    assert len(session.messages) == 0
    
    # Add messages
    message1 = session.add_message("user", "Hello")
    message2 = session.add_message("assistant", "Hi there")
    
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"
    
    # Get history
    history = session.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    
    # Get context window
    context = session.get_context_window(1)
    assert len(context) == 1
    assert context[0]["role"] == "assistant"
    
    # Clear session
    session.clear()
    assert len(session.messages) == 0


def test_data_chat_initialization(config):
    """Test DataChat initialization."""
    data_chat = DataChat(config)
    
    assert isinstance(data_chat, DataChat)
    assert data_chat.config == config
    assert data_chat.analyzer is not None
    assert data_chat.session is not None
    assert data_chat.current_df is None


def test_data_chat_set_dataframe(config, sample_df):
    """Test setting DataFrame in DataChat."""
    data_chat = DataChat(config)
    data_chat.set_dataframe(sample_df)
    
    assert data_chat.current_df is not None
    assert data_chat.current_df.equals(sample_df)


def test_data_chat_ask_no_dataframe(config):
    """Test asking a question without setting a DataFrame."""
    data_chat = DataChat(config)
    response = data_chat.ask("What are the missing values?")
    
    assert "Please provide a DataFrame to analyze" in response


def test_data_chat_ask_with_dataframe(config, sample_df, monkeypatch):
    """Test asking a question with a DataFrame set."""
    # Mock the analyzer's analyze method to avoid actual LLM calls
    def mock_analyze(*args, **kwargs):
        return {
            "success": True,
            "answer": "There are 10 missing values in the income column.",
            "raw_result": {
                "missing_counts": {"income": 10},
                "total_missing": 10,
                "missing_percentage": 10.0
            }
        }
    
    data_chat = DataChat(config)
    data_chat.analyzer.analyze = mock_analyze
    data_chat.set_dataframe(sample_df)
    
    response = data_chat.ask("What are the missing values?")
    
    assert "Missing Values Analysis" in response
    assert "income" in response
    
    # Check that the interaction was tracked
    history = data_chat.get_session_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_data_chat_visualization(config, sample_df, monkeypatch, tmpdir):
    """Test visualization request handling."""
    # Create a temporary plot file
    plot_path = tmpdir.join("test_plot.png")
    with open(plot_path, "w") as f:
        f.write("test plot content")
    
    # Mock the analyzer's analyze method to return a visualization result
    def mock_analyze(*args, **kwargs):
        return {
            "success": True,
            "answer": "Here's a visualization of the data.",
            "plot_type": "scatter",
            "columns": ["age", "income"],
            "plot_path": str(plot_path)
        }
    
    data_chat = DataChat(config)
    data_chat.analyzer.analyze = mock_analyze
    data_chat.set_dataframe(sample_df)
    
    response = data_chat.ask("Create a scatter plot of age vs income")
    
    assert "Data Visualization" in response
    assert "scatter" in response.lower()
    assert "age" in response
    assert "income" in response
    
    # Check that the plot path is in the response
    assert f"PLOT_PATH:{plot_path}" in response
    
    # Test get_last_plot_path
    last_plot_path = data_chat.get_last_plot_path()
    assert last_plot_path == str(plot_path)


def test_data_chat_error_handling(config, sample_df, monkeypatch):
    """Test error handling in DataChat."""
    # Mock the analyzer's analyze method to return an error
    def mock_analyze(*args, **kwargs):
        return {
            "success": False,
            "error": "Failed to analyze data",
            "error_type": "AnalysisError"
        }
    
    data_chat = DataChat(config)
    data_chat.analyzer.analyze = mock_analyze
    data_chat.set_dataframe(sample_df)
    
    response = data_chat.ask("What are the correlations?")
    
    assert "Sorry, I encountered an error" in response
    assert "Failed to analyze data" in response
    
    # Check that the error interaction was tracked
    history = data_chat.get_session_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert "Sorry, I encountered an error" in history[1]["content"] 