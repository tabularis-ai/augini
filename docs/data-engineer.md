# DataEngineer

The DataEngineer API provides powerful tools for feature engineering and data augmentation using AI.

## Quick Start

```python
from augini import DataEngineer
import pandas as pd

# Initialize with minimal configuration
engineer = DataEngineer(api_key="sk-...")

# Create sample data
df = pd.DataFrame({
    'CustomerID': ['C001', 'C002'],
    'Age': [25, 45],
    'MonthlyCharges': [50.0, 75.0]
})

# Generate a single feature
df = engineer.generate_feature(
    df=df,
    new_feature_name='customer_segment',
    new_feature_description='Classify customer into segments based on spending',
    output_type='category',
    constraints={'categories': ['Premium', 'Regular', 'Budget']}
)
```

## Feature Generation Examples

### Single Feature Generation

```python
# Generate occupation prediction
df = engineer.generate_feature(
    df=df,
    new_feature_name='PredictedOccupation',
    new_feature_description="Predict occupation based on age and spending",
    output_type='text'
)

# Example output:
#   CustomerID  Age  MonthlyCharges  PredictedOccupation
# 0      C001   25           50.0   Entry Level Professional
# 1      C002   45           75.0   Senior Manager
```

### Multiple Features Generation

```python
augmented_df = engineer.generate_features(
    df=df,
    features=[
        {
            'new_feature_name': 'ChurnRisk',
            'new_feature_description': 'Calculate churn risk score (0-100)',
            'output_type': 'float',
            'constraints': {'min': 0, 'max': 100}
        },
        {
            'new_feature_name': 'RetentionOffer',
            'new_feature_description': 'Suggest personalized retention offer',
            'output_type': 'text'
        }
    ],
    use_sync=False,  # Async processing for better performance
    show_progress=True
)

# Example output:
#   CustomerID  Age  MonthlyCharges  ChurnRisk  RetentionOffer
# 0      C001   25           50.0       35.5   "10% discount..."
# 1      C002   45           75.0       25.2   "Premium upgrade..."
```

## Advanced Configuration

```python
engineer = DataEngineer(
    api_key="sk-...",
    model="gpt-4o-mini",  # Model selection
    temperature=0.8,      # Response creativity
    max_tokens=750,       # Max response length
    concurrency_limit=10, # Parallel processing limit
    base_url="https://openrouter.ai/api/v1",  # Custom API endpoint
    debug=False          # Debug mode
)
```

## Output Types and Constraints

```python
# Numeric output with constraints
df = engineer.generate_feature(
    df=df,
    new_feature_name='risk_score',
    new_feature_description='Calculate risk score',
    output_type='float',
    constraints={
        'min': 0,
        'max': 100,
        'step': 0.1
    }
)

# Categorical output with defined categories
df = engineer.generate_feature(
    df=df,
    new_feature_name='segment',
    new_feature_description='Customer segment',
    output_type='category',
    constraints={
        'categories': ['Premium', 'Regular', 'Budget']
    }
)
```

## Key Features

- AI-powered feature generation
- Automated feature engineering
- Multiple output types support (float, category, text)
- Batch processing capabilities
- Custom constraints and validations

## Basic Usage

```python
from augini import DataEngineer
import pandas as pd

# Initialize with configuration
config = {
    'api_key': 'your-api-key',
    'model': 'gpt-4-turbo-preview'
}

engineer = DataEngineer(config=config)

# Load your data
data = pd.read_csv('your_data.csv')

# Generate a new feature
result = engineer.generate_feature(
    df=data,
    new_feature_name='risk_score',
    new_feature_description='Calculate customer risk score based on transaction history',
    output_type='float'
)
```

## Feature Generation

### Single Feature Generation

```python
# Generate a specific feature
feature_data = engineer.generate_feature(
    df=data,
    new_feature_name='customer_segment',
    new_feature_description="Create customer segments based on behavior",
    output_type='category',
    source_columns=['age', 'income', 'purchase_history']
)
```

### Multiple Features Generation

```python
# Generate multiple features at once
features_data = engineer.generate_features(
    df=data,
    features=[
        {
            'new_feature_name': 'lifetime_value',
            'new_feature_description': "Predict customer lifetime value",
            'output_type': 'float',
            'constraints': {'min': 0}
        },
        {
            'new_feature_name': 'churn_risk',
            'new_feature_description': "Assess customer churn risk",
            'output_type': 'category'
        }
    ]
)
```

## Advanced Usage

### Custom Constraints

```python
# Generate feature with constraints
result = engineer.generate_feature(
    df=data,
    new_feature_name='satisfaction_score',
    new_feature_description='Calculate customer satisfaction score',
    output_type='float',
    constraints={
        'min': 0,
        'max': 100,
        'step': 0.1
    }
)
```

### Batch Processing

```python
# Handle large datasets efficiently
result = engineer.generate_feature(
    df=large_data,
    new_feature_name='risk_score',
    new_feature_description='Calculate risk score',
    output_type='float',
    batch_size=32
)
```

## Configuration Options

```python
config = {
    # Model settings
    'model': 'gpt-4o-mini',
    'temperature': 0.7,
    
    # Processing settings
    'batch_size': 32,
    'concurrency_limit': 5,
    
    # Performance settings
    'enable_cache': True,
    'cache_ttl': 3600
}
```

## Best Practices

1. Start with a small subset of data to test feature generation
2. Use descriptive feature names and clear descriptions
3. Specify appropriate output types and constraints
4. Use batch processing for large datasets
5. Monitor and validate generated features before production use 