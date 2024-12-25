[![PyPI version](https://badge.fury.io/py/augini.svg)](https://badge.fury.io/py/augini) [![Downloads](https://static.pepy.tech/badge/augini)](https://pepy.tech/project/augini)

# augini: AI-Powered Data Assistant

<p align="center">
  <img src="img/logo_augini.png" alt="augini logo" width="200"/>
</p>

`augini` is an AI-powered data assistant that helps you understand, augment, and transform your tabular data. Built with state-of-the-art language models, it provides an intuitive chat interface and powerful data manipulation capabilities.

## Key Features

### 🤖 Interactive Data Chat

Have natural conversations with your data using `augini`'s chat interface:

```python
from augini import Augini
import pandas as pd

# Initialize with your preferred model
augini = Augini(api_key="your-api-key", model="gpt-4o-mini")

# Load your data
df = pd.read_csv("your_data.csv")

# Start chatting with your data - properly display markdown responses
from IPython.display import display, Markdown

response = augini.chat("What are the main patterns in this dataset?", df)
display(Markdown(response))

# Ask follow-up questions
response = augini.chat("Can you analyze the correlation between age and income?", df)
display(Markdown(response))
```

### 🔄 Intelligent Data Augmentation

Enhance your datasets with AI-generated features:

```python
# Add synthetic features based on existing data
result_df = augini.augment_columns(df, ['occupation', 'interests', 'personality_type'])

# Generate custom features with specific prompts
custom_prompt = """
Based on the person's age and location, suggest:
1. A likely income bracket
2. Preferred shopping categories
3. Travel preferences

Respond with a JSON object with keys 'income_bracket', 'shopping_preferences', 'travel_style'.
"""

enriched_df = augini.augment_columns(df, 
    ['income_bracket', 'shopping_preferences', 'travel_style'],
    custom_prompt=custom_prompt
)
```

### 🔒 Data Anonymization

Generate privacy-safe synthetic data while preserving statistical properties:

```python
# Define anonymization strategy
anonymize_prompt = """
Create an anonymized version that:
1. Replaces personal identifiers with synthetic data
2. Maintains statistical distributions
3. Preserves relationships between variables

Respond with a JSON object containing anonymized values.
"""

# Apply anonymization
anonymous_df = augini.augment_columns(df, 
    ['name_anon', 'email_anon', 'address_anon'],
    custom_prompt=anonymize_prompt
)
```

## Installation

```bash
pip install augini
```

## Quick Start

1. Get your API key from OpenAI or OpenRouter
2. Initialize Augini:
```python
# Using OpenAI
augini = Augini(api_key="your-api-key", model="gpt-4o-mini", use_openrouter=False)

# Using OpenRouter
augini = Augini(api_key="your-api-key", model="meta-llama/llama-3-8b-instruct", use_openrouter=True)
```

## Enterprise Solutions

For enterprise deployments, local installations, or custom solutions, contact us:
- Email: info@tabularis.ai