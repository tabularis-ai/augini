# Chat Interface

The Chat interface provides an interactive way to analyze and understand your data through natural language conversations.

## Key Features

- Natural language data analysis
- Context-aware responses
- Memory of previous interactions
- Custom analysis queries
- Interactive visualizations

## Basic Usage

```python
from augini import Chat
import pandas as pd

# Initialize chat interface
chat = Chat(config={
    'api_key': 'your-api-key',
    'model': 'gpt-4-turbo-preview'
})

# Load your data
data = pd.read_csv('your_data.csv')

# Start chatting about your data
response = chat.ask(
    data,
    "What are the main trends in this dataset?"
)
```

## Advanced Features

### Context Management

```python
# Enable context awareness
chat = Chat(config={'enable_memory': True})

# Chat maintains context across questions
response1 = chat.ask(data, "What's the average age?")
response2 = chat.ask(data, "How does it correlate with income?")
```

### Custom Analysis

```python
# Perform specific analyses through chat
analysis = chat.ask(
    data,
    "Create a cohort analysis based on signup date",
    output_format='dataframe'
)
```

### Visualization Requests

```python
# Generate visualizations
plot = chat.ask(
    data,
    "Show me a trend plot of sales over time",
    output_format='plot'
)
```

## Configuration Options

```python
config = {
    # Chat settings
    'enable_memory': True,
    'memory_window': 10,  # Remember last 10 interactions
    
    # Response settings
    'response_format': 'markdown',
    'include_visualizations': True,
    
    # Model settings
    'temperature': 0.7,
    'max_tokens': 1000
}
```

## Best Practices

1. Ask clear, specific questions
2. Use context to build complex analyses
3. Specify output formats when needed
4. Review and validate generated insights
5. Save important analyses for future reference 