# Augini 🤖

<p align="center">
  <img src="docs/assets/images/logo_augini.png" alt="augini logo" width="200"/>
</p>

<div align="center">
  
[![PyPI version](https://badge.fury.io/py/augini.svg)](https://badge.fury.io/py/augini) 
[![Downloads](https://static.pepy.tech/badge/augini)](https://pepy.tech/project/augini)
[![Documentation](https://img.shields.io/badge/docs-augini-blue)](https://tabularis-ai.github.io/augini/)
[![Discord](https://img.shields.io/discord/1310217643520819251?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1310217643520819251/)
[![Twitter Follow](https://img.shields.io/twitter/follow/tabularis_ai?style=social)](https://x.com/tabularis_ai)
![Last Commit](https://img.shields.io/github/last-commit/tabularis-ai/augini)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-white?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/tabularisai)

</div>

## 🎯 What is Augini?

Augini is an AI-powered Python framework for tabular data enrichment and analysis. It leverages Large Language Models (LLMs) to:
- Generate meaningful features from your data
- Provide natural language data analysis
- Create AI agents for automated data workflows

## 🚀 Quick Start

```bash
pip install augini
```

```python
from augini import DataEngineer, DataAnalyzer
import pandas as pd

# Sample customer data
df = pd.DataFrame({
    'CustomerID': ['C001', 'C002'],
    'Age': [25, 45],
    'MonthlyCharges': [50.0, 75.0]
})

# Initialize with your API key (supports OpenAI, OpenRouter, Azure)
engineer = DataEngineer(
    api_key="your-api-key",
    model="gpt-4o-mini",  # Use OpenRouter's GPT-4
    base_url="https://openrouter.ai/api/v1"  # Optional: use OpenRouter
)

# Generate customer insights
df = engineer.generate_features(
    df=df,
    new_feature_specs=[
        {
            'new_feature_name': 'CustomerSegment',
            'new_feature_description': 'Classify customer segment based on age and spending',
            'output_type': 'category',
            'constraints': {'categories': ['Premium', 'Regular', 'Budget']}
        },
        {
            'new_feature_name': 'ChurnRisk',
            'new_feature_description': 'Calculate churn risk score (0-100)',
            'output_type': 'float',
            'constraints': {'min': 0, 'max': 100}
        }
    ]
)

# Initialize analyzer for data analysis
analyzer = DataAnalyzer(
    api_key="your-api-key",
    model="gpt-4o-mini"
)

# Analyze data
missing_values = analyzer.analyze_missing_values(df)
correlations = analyzer.analyze_correlations(df)
statistics = analyzer.analyze_statistics(df)
visualization = analyzer.analyze_visualizations(
    df=df,
    plot_type="scatter",
    x="Age",
    y="MonthlyCharges",
    hue="CustomerSegment"
)
```

## 🎁 Key Features

### 🔄 DataEngineer
- **Feature Generation**: Create meaningful features using AI
- **Data Augmentation**: Enrich datasets with synthetic data
- **Custom Constraints**: Control output formats and ranges
- **Batch Processing**: Handle large datasets efficiently

### 📊 DataAnalyzer
- **Statistical Analysis**: Get insights about your data
- **Pattern Detection**: Uncover hidden trends and correlations
- **Visualization Integration**: Generate plots and charts
- **Tool-based Architecture**: Modular and extensible analysis

## 🌐 Provider Agnostic

Augini works with multiple LLM providers:
- OpenAI
- OpenRouter
- Azure OpenAI
- Anthropic (coming soon)


## 🤝 Contributing

We welcome contributions! 

## 📜 License

Augini is released under the [MIT License](LICENSE).

# Augini Data Chat

A conversational interface for data analysis using the Augini framework.

## Overview

The DataChat class provides a simple chat interface for interacting with the DataAnalyzer. It allows users to ask questions about their data in natural language and receive formatted responses with markdown support.

## Features

- Simple chat interface for data analysis
- Support for follow-up questions with context from previous interactions
- Visualization request handling with automatic plot generation
- Markdown-formatted responses for better readability
- Session tracking for conversation history
- Error handling with helpful suggestions
- Integration with the Augini configuration system

## Usage

### API Key Setup

Before using Augini, you need to set up an API key for the LLM provider:

```bash
# For OpenAI
export OPENAI_API_KEY=your-openai-api-key

# For OpenRouter
export OPENROUTER_TOKEN=your-openrouter-token
```

### Basic Usage

```python
import pandas as pd
from augini import DataChat, AuginiConfig

# Load or create configuration
config = AuginiConfig.from_env()  # Load from environment variables
# or
# config = AuginiConfig.from_file("config.yaml")  # Load from file

# Create DataChat instance
data_chat = DataChat(config)

# Load your DataFrame
df = pd.read_csv("your_data.csv")
data_chat.set_dataframe(df)

# Ask a question (simplified interface)
response = data_chat.ask("What are the missing values in this dataset?")
print(response)

# Ask a follow-up question (context is maintained)
response = data_chat.ask("Can you visualize the distribution of column X?")
print(response)

# Get the chat history
history = data_chat.get_session_history()
```

### Visualization Support

DataChat supports generating and displaying visualizations:

```python
# Ask for a visualization
response = data_chat.ask("Create a scatter plot of age vs income")
print(response)

# Display the generated plot (if any)
plot_path = data_chat.get_last_plot_path()
if plot_path:
    # For Jupyter notebooks
    from IPython.display import Image, display
    display(Image(plot_path))
    
    # Or for regular Python scripts
    import matplotlib.pyplot as plt
    img = plt.imread(plot_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
```

### Example Scripts

See the following example scripts for complete examples of how to use the DataChat class:

- `examples/data_chat_example.py` - Interactive chat example with a command-line interface
- `examples/simple_chat.py` - Example using the simplified interface with visualization support

## Configuration

The DataChat class uses the Augini configuration system. You can configure the following aspects:

- LLM provider and parameters
- Agent settings (max iterations, early stopping, etc.)
- Tool configurations (missing values analyzer, correlation analyzer, etc.)

Example configuration:

```yaml
llm:
  provider: openrouter
  api_key: your-api-key
  model: anthropic/claude-3-sonnet
  temperature: 0.7
  max_tokens: 1000

agent:
  max_iterations: 5
  early_stopping: true
  verbose: false
  return_intermediate_steps: false

tools:
  missing_values_analyzer:
    enabled: true
  correlation_analyzer:
    enabled: true
  statistical_summary:
    enabled: true
  visualization_tool:
    enabled: true
```

## License

[MIT License](LICENSE)