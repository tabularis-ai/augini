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
    features=[
        {
            'name': 'CustomerSegment',
            'description': 'Classify customer segment based on age and spending',
            'output_type': 'category',
            'constraints': {'categories': ['Premium', 'Regular', 'Budget']}
        },
        {
            'name': 'ChurnRisk',
            'description': 'Calculate churn risk score (0-100)',
            'output_type': 'float',
            'constraints': {'min': 0, 'max': 100}
        }
    ]
)

# Initialize analyzer for natural language insights
analyzer = DataAnalyzer(
    api_key="your-api-key",
    model="gpt-4o-mini",
    enable_memory=True  # Enable conversation context
)

# Fit data and ask questions
analyzer.fit(df)
insights = analyzer.chat("What patterns do you see in customer segments?")
print(insights)
```

## 🎁 Key Features

### 🔄 DataEngineer
- **Feature Generation**: Create meaningful features using AI
- **Data Augmentation**: Enrich datasets with synthetic data
- **Custom Constraints**: Control output formats and ranges
- **Batch Processing**: Handle large datasets efficiently

### 📊 DataAnalyzer
- **Natural Language Analysis**: Ask questions about your data
- **Pattern Detection**: Uncover hidden trends and correlations
- **Memory Context**: Build on previous analysis
- **Visualization Integration**: Generate plots and charts

### 🤖 AI Agents
- **Automated Workflows**: Create agents for repetitive tasks
- **Custom Behaviors**: Define agent goals and constraints
- **Chain Actions**: Connect multiple agents for complex workflows

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