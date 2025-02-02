# Provider Agnostic Usage

Augini is designed to work with different LLM providers. Here's how to configure it for various providers.

## OpenRouter Integration

```python
from augini import DataEngineer, DataAnalyzer

# Configure for OpenRouter
engineer = DataEngineer(
    api_key="your-openrouter-key",
    model="gpt-4o-mini",  # OpenRouter model name
    base_url="https://openrouter.ai/api/v1",
    temperature=0.8,
    max_tokens=750
)

# Example output:
# Using OpenRouter API endpoint...
# Model set to gpt-4o-mini
```

## OpenAI Direct Integration

```python
# Configure for OpenAI
engineer = DataEngineer(
    api_key="sk-...",  # Your OpenAI API key
    model="gpt-4-turbo-preview",
    temperature=0.8,
    max_tokens=750
)

# Example output:
# Using OpenAI API endpoint...
# Model set to gpt-4-turbo-preview
```

## Azure OpenAI Integration

```python
# Configure for Azure OpenAI
engineer = DataEngineer(
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com",
    model="gpt-4",
    api_version="2024-02-15-preview"
)
```

## Configuration Options

Common configuration parameters across providers:

```python
config = {
    'api_key': str,        # API key for authentication
    'base_url': str,       # API endpoint URL
    'model': str,          # Model identifier
    'temperature': float,  # Response creativity (0.0-1.0)
    'max_tokens': int,    # Maximum response length
    'timeout': int,       # Request timeout in seconds
}
``` 