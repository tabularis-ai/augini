# Augini

Augini is a Python framework for generating synthetic tabular data using AI. It leverages the power of language models to create realistic, fictional data based on existing datasets.

## Installation

You can install Augini using pip:
```sh
pip install augini
```

## Quick Start

Here's a simple example of how to use Augini:

```python
from augini import Augini
import pandas as pd

# Initialize Augini
augini = Augini(api_key="your_api_key", use_openrouter=True)

# Create a sample DataFrame
data = {
    'Place of Birth': ['New York', 'London', 'Tokyo'],
    'Age': [30, 25, 40],
    'Gender': ['Male', 'Female', 'Male']
}
df = pd.DataFrame(data)

# Add synthetic features
result_df = augini.augment_columns(df, 'NAME', 'OCCUPATION', 'FAVORITE_DRINK')

print(result_df)
```

## Features

- Generate synthetic data based on existing datasets
- Customizable prompts for data generation
- Support for both OpenAI API and OpenRouter
- Asynchronous processing for improved performance

## Extending and Enriching Data

Augini can be used to extend, augment, and enrich your datasets by adding synthetic features and bringing knowledge from language models to your data.

### Adding Multiple Features

You can add multiple features to your DataFrame:

```python
result_df = augini.augment_columns(df, 'Hobby', 'FavoriteColor', 'FavoriteMovie')
print(result_df)
```

### Custom Prompts for Feature Generation

Custom prompts allow you to generate specific features based on your needs:

```python
custom_prompt = "Based on the person's name and age, suggest a quirky pet for them. Respond with a JSON object with the key 'QuirkyPet'."
result_df = augini.augment_single(df, 'QuirkyPet', custom_prompt=custom_prompt)
print(result_df)
```

### Anonymizing Data

You can anonymize sensitive information in your dataset by generating synthetic data:

```python
anonymize_prompt = "Create an anonymous profile for the person based on their age and city. Respond with a JSON object with keys 'AnonymousName' and 'AnonymousEmail'."
result_df = augini.augment_single(df, 'AnonymousProfile', custom_prompt=anonymize_prompt)
print(result_df)
```

## Bringing Knowledge from LLMs

Leverage the knowledge embedded in language models to enhance your datasets:

### Generating Detailed Descriptions

```python
description_prompt = "Generate a detailed description for a person based on their age and city. Respond with a JSON object with the key 'Description'."
result_df = augini.augment_single(df, 'Description', custom_prompt=description_prompt)
print(result_df)
```

### Suggesting Recommendations

```python
recommendation_prompt = "Suggest a book and a movie for a person based on their age and city. Respond with a JSON object with keys 'RecommendedBook' and 'RecommendedMovie'."
result_df = augini.augment_single(df, 'Recommendations', custom_prompt=recommendation_prompt)
print(result_df)
```

## Full Example

Here's a full example demonstrating multiple features and custom prompts:

```python
from augini import Augini
import pandas as pd

# Initialize Augini
augini = Augini(api_key="your_api_key", use_openrouter=True)

# Create a sample DataFrame
data = {
    'Name': ['Alice Johnson', 'Bob Smith', 'Charlie Davis'],
    'Age': [28, 34, 45],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# Add multiple synthetic features
result_df = augini.augment_columns(df, 'Occupation', 'Hobby', 'FavoriteColor')

# Add a custom feature
custom_prompt = "Based on the person's name and age, suggest a quirky pet for them. Respond with a JSON object with the key 'QuirkyPet'."
result_df = augini.augment_single(result_df, 'QuirkyPet', custom_prompt=custom_prompt)

# Anonymize data
anonymize_prompt = "Create an anonymous profile for the person based on their age and city. Respond with a JSON object with keys 'AnonymousName' and 'AnonymousEmail'."
result_df = augini.augment_single(result_df, 'AnonymousProfile', custom_prompt=anonymize_prompt)

print(result_df)
```

## Contributing

We welcome contributions to enhance Augini! Feel free to open issues and submit pull requests on our GitHub repository.
