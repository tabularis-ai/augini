# Quick Start Guide

This guide will help you get started with Augini quickly using a practical example.

## Installation

```bash
pip install augini
```

## Basic Configuration

Configure Augini using environment variables or a configuration file:

```python
from augini.config import AuginiConfig

# Using environment variables
config = AuginiConfig.from_env()

# Or using a YAML file
config = AuginiConfig.from_file('config.yaml')
```

Example `config.yaml`:
```yaml
api_key: your-api-key
model: gpt-4-turbo-preview
```

## Practical Example

Let's walk through a complete example using a sample customer dataset:

```python
from augini import DataEngineer, DataAnalyzer
import pandas as pd

# Create a sample customer dataset
data = pd.DataFrame({
    'customer_id': range(1, 6),
    'age': [25, 35, 45, 28, 52],
    'income': [30000, 45000, 75000, 35000, 85000],
    'purchase_amount': [150, 450, 850, 250, 950],
    'location': ['NY', 'CA', 'TX', 'FL', 'WA']
})

# Initialize Augini components
engineer = DataEngineer(api_key='your-api-key')
analyzer = DataAnalyzer(api_key='your-api-key')

# Step 1: Generate new features using DataEngineer
engineered_data = engineer.generate_feature(
    df=data,
    new_feature_name='customer_segment',
    new_feature_description="Create customer segments based on age, income, and purchase_amount",
    output_type='category'
)

print("Generated Features:")
print(engineered_data[['customer_id', 'customer_segment']])

# Step 2: Analyze the data using DataAnalyzer
analyzer.fit(engineered_data)  # Prepare the analyzer with our data
insights = analyzer.chat(
    "What are the key characteristics of each customer segment? "
    "Focus on average age, income, and purchase amounts."
)

print("\nSegment Analysis:")
print(insights)

# Step 3: Ask follow-up questions
follow_up = analyzer.chat(
    "Which segment shows the highest potential for growth?",
    use_memory=True  # Use context from previous question
)

print("\nFollow-up Analysis:")
print(follow_up)
```

This example demonstrates:
1. Creating a sample dataset
2. Using DataEngineer to add customer segmentation
3. Using DataAnalyzer to understand segments through natural language
4. Asking follow-up questions with context memory

## Common Operations

### Feature Generation
```python
# Generate a single feature
feature_data = engineer.generate_feature(
    df=data,
    new_feature_name='risk_score',
    new_feature_description="Generate risk score based on customer behavior",
    output_type='float'
)

# Generate multiple features
features_data = engineer.generate_features(
    df=data,
    features=[
        {
            'new_feature_name': 'lifetime_value',
            'new_feature_description': "Predict customer lifetime value",
            'output_type': 'float'
        },
        {
            'new_feature_name': 'churn_risk',
            'new_feature_description': "Assess customer churn risk",
            'output_type': 'category'
        }
    ]
)
```

### Interactive Analysis
```python
# First prepare the analyzer
analyzer.fit(data)

# Ask questions about your data
basic_insights = analyzer.chat(
    "What are the main patterns in purchase behavior?"
)

# Ask follow-up questions
detailed_insights = analyzer.chat(
    "How do these patterns vary by age group?",
    use_memory=True
)
```

For more detailed information about each component, check the API documentation:
- [DataEngineer API](data-engineer.md)
- [DataAnalyzer API](data-analyzer.md)
- [Chat Interface](chat.md)

For more detailed information, check the [API Reference](api-reference.md) or [Advanced Topics](advanced.md). 