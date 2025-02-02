# DataAnalyzer

The DataAnalyzer API provides an AI-powered interface for interactive data analysis through natural language queries.

## Key Features

- Natural language data analysis
- Interactive chat interface
- Context-aware responses
- Statistical insights
- Pattern detection
- Trend analysis

## Basic Usage

```python
from augini import DataAnalyzer
import pandas as pd

# Initialize with configuration
config = {
    'api_key': 'your-api-key',
    'model': 'gpt-4-turbo-preview'
}

analyzer = DataAnalyzer(config=config)

# Load your data and prepare analyzer
data = pd.read_csv('your_data.csv')
analyzer.fit(data)

# Ask questions about your data
insights = analyzer.chat("What are the main trends in this dataset?")
print(insights)
```

## Analysis Types

### Statistical Analysis

```python
# Ask about statistical patterns
stats = analyzer.chat(
    "What are the key statistical patterns in the data? "
    "Include mean, median, and correlations in your analysis."
)
```

### Pattern Detection

```python
# Ask about patterns in time series
patterns = analyzer.chat(
    "What patterns do you see in the data over time? "
    "Focus on the 'date' column."
)
```

### Trend Analysis

```python
# Ask about trends by category
trends = analyzer.chat(
    "How do metrics vary across different categories? "
    "Group the analysis by 'category' column."
)
```

## Advanced Usage

### Memory Features

```python
# Enable conversation memory for context-aware analysis
analyzer = DataAnalyzer(
    api_key='your-api-key',
    enable_memory=True
)
analyzer.fit(data)

# First question
response1 = analyzer.chat(
    "What's the average age in the dataset?",
    use_memory=True
)

# Follow-up question (uses context from previous question)
response2 = analyzer.chat(
    "How does it correlate with income?",
    use_memory=True
)
```

### Custom Analysis

```python
# Ask specific analytical questions
analysis = analyzer.chat(
    "Create a cohort analysis based on signup date. "
    "Show retention rates over time and identify key patterns."
)
```

## Configuration Options

```python
config = {
    # Model settings
    'model': 'gpt-4-turbo-preview',
    'temperature': 0.7,
    
    # Memory settings
    'enable_memory': True,
    'context_window_tokens': 1000,
    
    # Debug settings
    'debug': True,
    'log_level': 'INFO'
}
```

## Best Practices

1. Always call fit() before chat()
2. Ask clear, specific questions
3. Use memory features for related queries
4. Provide context in your questions
5. Validate insights against raw data 