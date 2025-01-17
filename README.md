[![PyPI version](https://badge.fury.io/py/augini.svg)](https://badge.fury.io/py/augini) [![Downloads](https://static.pepy.tech/badge/augini)](https://pepy.tech/project/augini)

# augini: AI-Powered Tabular Data Assistant

🔥  Demo: https://huggingface.co/spaces/tabularisai/augini

<p align="center">
  <img src="img/logo_augini.png" alt="augini logo" width="200"/>
</p>

`augini` is an AI-powered data assistant that brings RAG (Retrieval-Augmented Generation) capabilities to your tabular data (CSV, Excel, XLSX). Built with state-of-the-art language models, it provides an intuitive chat interface for data analysis and powerful data manipulation capabilities.

## Key Features

### 🤖 Interactive Data Chat (aka RAG for Tables)

Have natural conversations with your data using `augini`'s chat interface. Works with any tabular format (CSV, Excel, Pandas DataFrames):

```python
import augini as au
import pandas as pd

# Load your data (CSV, Excel, or any pandas-supported format)
df = pd.read_csv("your_data.csv")  # or pd.read_excel("your_data.xlsx")

# Initialize the chat interface
chat = au.Chat(
    df=df,
    api_key="your-api-key",
    model="gpt-4o-mini",
    use_openrouter=True, # use openrouter api if False then use openai api
)

# Start chatting with your data - properly display markdown responses
from IPython.display import display, Markdown

response = chat("What are the main patterns in this dataset?")
display(Markdown(response))

# Ask follow-up questions with context awareness
response = chat("Can you analyze the correlation between age and income?")
display(Markdown(response))
```

### 🔄 Intelligent Data Augmentation

Enhance your datasets with AI-generated features:

```python
import augini as au
import pandas as pd

# Initialize the augmentation interface
augmenter = au.Augment(
    api_key="your-api-key",
    model="gpt-4o-mini"
)

# Add synthetic features based on existing data
result_df = augmenter.augment_columns(df, ['occupation', 'interests', 'personality_type'])

# Generate custom features with specific prompts
custom_prompt = """
Based on the person's age and location, suggest:
1. A likely income bracket
2. Preferred shopping categories
3. Travel preferences

Respond with a JSON object with keys 'income_bracket', 'shopping_preferences', 'travel_style'.
"""

enriched_df = augmenter.augment_columns(df, 
    columns=['income_bracket', 'shopping_preferences', 'travel_style'],
    custom_prompt=custom_prompt
)

# Generate single column
result_df = augmenter.augment_single(df, 
    column_name='occupation',
    custom_prompt="Suggest a realistic occupation based on the person's profile."
)
```

### 🔒 Data Anonymization

Generate privacy-safe synthetic data while preserving statistical properties:

```python
import augini as au
import pandas as pd

# Define anonymization strategy
anonymize_prompt = """
Create an anonymized version that:
1. Replaces personal identifiers with synthetic data
2. Maintains statistical distributions
3. Preserves relationships between variables

Respond with a JSON object containing anonymized values.
"""

# Apply anonymization
anonymous_df = augmenter.augment_columns(df, 
    columns=['name_anon', 'email_anon', 'address_anon'],
    custom_prompt=anonymize_prompt
)
```

## Installation

```bash
pip install augini
```

## Quick Start

1. Get your API key from OpenAI or OpenRouter
2. Initialize augini components:
```python
import augini as au
import pandas as pd

# Using OpenAI
chat = au.Chat(df=df, api_key="your-api-key", model="gpt-4", use_openrouter=False)
augmenter = au.Augment(api_key="your-api-key", model="gpt-4", use_openrouter=False)

# Using OpenRouter
chat = au.Chat(df=df, api_key="your-api-key", model="meta-llama/llama-3-8b-instruct", use_openrouter=True)
augmenter = au.Augment(api_key="your-api-key", model="meta-llama/llama-3-8b-instruct", use_openrouter=True)
```

## Advanced Features

### Memory Management in Chat

The Chat interface supports conversation memory for better context awareness:

```python
import augini as au
import pandas as pd

# Enable memory during initialization
chat = au.Chat(df=df, api_key="your-api-key", enable_memory=True)

# Access and manage conversation history
full_history = chat.get_conversation_history('full')
summary_history = chat.get_conversation_history('summary')

# Clear history when needed
chat.clear_conversation_history('all')  # or 'full' or 'summary'
```

### Concurrent Data Augmentation

The Augment interface supports efficient concurrent processing:

```python
import augini as au

augmenter = au.Augment(
    api_key="your-api-key",
    max_concurrent=5  # Control concurrent API calls
)
```

## Enterprise Solutions

For enterprise deployments, local installations, or custom solutions, contact us:
- Email: info@tabularis.ai

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/Discord%20button.png" width="200"/>](https://discord.gg/sznxwdqBXj)