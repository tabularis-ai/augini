#!/usr/bin/env python3
"""
Simple example demonstrating the simplified DataChat usage with visualization support.
"""

# Standard library imports
import os
import sys
import traceback

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Augini imports
from augini import (
    DataChat, 
    AuginiConfig, 
    LLMConfig, 
    AgentConfig, 
    LLMProvider, 
    ToolConfig
)

def create_sample_data():
    """Create a sample DataFrame for demonstration."""
    np.random.seed(42)
    
    # Create a DataFrame with some missing values
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.randint(20000, 150000, 100),
        'education_years': np.random.randint(8, 22, 100),
        'satisfaction': np.random.randint(1, 11, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    
    # Add some missing values
    for col in df.columns:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    
    # Add some correlations
    df['income'] = df['education_years'] * 5000 + np.random.normal(0, 10000, 100)
    
    return df

def create_config():
    """Create a configuration for the DataChat."""
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_TOKEN")
    
    if not api_key:
        print("Warning: No API key found in environment variables.")
        print("Please set OPENAI_API_KEY or OPENROUTER_TOKEN environment variable.")
        api_key = "your-api-key"  # Placeholder
    
    # Create LLM configuration
    llm_config = LLMConfig(
        provider=LLMProvider.OPENROUTER,
        api_key=api_key,
        model="anthropic/claude-3-sonnet",
        temperature=0.7,
        max_tokens=1000
    )
    
    # Create agent configuration
    agent_config = AgentConfig(
        max_iterations=5,
        early_stopping=True,
        verbose=True,
        return_intermediate_steps=True
    )
    
    # Create tool configurations
    tools = {
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
    }
    
    # Create the full configuration
    config = AuginiConfig(
        llm=llm_config,
        agent=agent_config,
        tools=tools,
        debug=True
    )
    
    return config

def display_plot(data_chat, response):
    """Display a plot if one was generated in the response.
    
    Args:
        data_chat: DataChat instance
        response: Response from the chat
    """
    # Check if there's a plot to display
    plot_path = data_chat.get_last_plot_path()
    
    if plot_path and os.path.exists(plot_path):
        try:
            # For Jupyter notebooks
            if 'IPython' in sys.modules and 'get_ipython' in dir():
                from IPython.display import Image, display
                display(Image(plot_path))
                print(f"Plot displayed in notebook from: {plot_path}")
            else:
                # For terminal/GUI applications
                try:
                    img = plt.imread(plot_path)
                    plt.figure(figsize=(10, 6))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Plot from: {os.path.basename(plot_path)}")
                    plt.show()
                    print(f"Plot displayed from: {plot_path}")
                except Exception as e:
                    print(f"Could not display plot with matplotlib: {str(e)}")
                    print(f"Plot saved at: {plot_path}")
                    # Try to open with system default application
                    try:
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(plot_path)}")
                        print(f"Attempted to open plot with default application")
                    except:
                        pass
        except Exception as e:
            print(f"Error displaying plot: {str(e)}")
            print(f"Plot saved at: {plot_path}")
    
    return response

def main():
    """Run the simple DataChat example with the simplified interface."""
    print("Simplified DataChat Example with Visualization")
    print("--------------------------------------------")
    
    try:
        # Create configuration
        config = create_config()
        print("✅ Configuration created successfully")
        
        # Check if we have a valid API key
        if config.llm.api_key == "your-api-key":
            print("⚠️  Warning: No API key found in environment variables.")
            print("Please set OPENAI_API_KEY or OPENROUTER_TOKEN environment variable.")
            print("Exiting example as API calls will fail without a valid key.")
            return
        
        # Create the DataChat instance
        data_chat = DataChat(config)
        print("✅ DataChat instance created successfully")
        
        # Load sample data
        df = create_sample_data()
        print("✅ Sample data created successfully")
        print(f"   DataFrame shape: {df.shape}")
        print(f"   DataFrame columns: {df.columns.tolist()}")
        
        # Set the DataFrame
        data_chat.set_dataframe(df)
        
        # Ask questions using the simplified interface
        print("\nAsking questions using the simplified interface:")
        
        # Question about missing values
        question1 = "How many missing values are there in each column?"
        print(f"\nQuestion: {question1}")
        answer1 = data_chat.ask(question1)
        print(f"Answer: {answer1}")
        
        # Question about statistics
        question2 = "What are the basic statistics of the numerical columns?"
        print(f"\nQuestion: {question2}")
        answer2 = data_chat.ask(question2)
        print(f"Answer: {answer2}")
        
        # Question about correlations
        question3 = "What are the correlations between education years and income?"
        print(f"\nQuestion: {question3}")
        answer3 = data_chat.ask(question3)
        print(f"Answer: {answer3}")
        
        # Question with visualization request
        question4 = "Can you create a scatter plot of education years vs income?"
        print(f"\nQuestion: {question4}")
        answer4 = data_chat.ask(question4)
        print(f"Answer: {answer4}")
        display_plot(data_chat, answer4)
        
        # Another visualization question
        question5 = "Show me a histogram of the age distribution"
        print(f"\nQuestion: {question5}")
        answer5 = data_chat.ask(question5)
        print(f"Answer: {answer5}")
        display_plot(data_chat, answer5)
        
        # Print the session history
        print("\nChat History:")
        for message in data_chat.get_session_history():
            print(f"{message['role'].capitalize()}: {message['content']}")
        
        print("\n✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 