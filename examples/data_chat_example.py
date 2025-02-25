#!/usr/bin/env python3
"""
Example script demonstrating how to use the DataChat interface.
"""

import pandas as pd
import numpy as np
import os
from augini.config.base import AuginiConfig, ToolConfig, LLMConfig, AgentConfig, LLMProvider
from augini.agents import DataChat

# Create sample data
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

def main():
    """Run the DataChat example."""
    # Create configuration
    config = create_config()
    
    # Create the DataChat instance
    data_chat = DataChat(config)
    
    # Set the DataFrame
    df = create_sample_data()
    data_chat.set_dataframe(df)
    
    print("DataChat Example")
    print("----------------")
    print("Type 'exit' to quit the chat.")
    print()
    
    # Simple chat loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        # Process the user's message
        response = data_chat.chat(user_input)
        
        print(f"Assistant: {response}")
        print()
    
    # Print the session history
    print("\nChat History:")
    for message in data_chat.get_session_history():
        print(f"{message['role'].capitalize()} ({message['timestamp']}): {message['content']}")

if __name__ == "__main__":
    main() 