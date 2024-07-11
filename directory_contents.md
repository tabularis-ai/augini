## README.md

```
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

```

## files.py

```
import os
import argparse

def process_directory(directory_path, output_file, extensions):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Check if file has one of the specified extensions
                if extensions and not any(file.endswith(ext) for ext in extensions):
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory_path)
                
                # Write the file path as a Markdown header
                outfile.write(f"## {relative_path}\n\n")
                
                # Write the file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write("```\n")
                        outfile.write(content)
                        outfile.write("\n```\n\n")
                except UnicodeDecodeError:
                    outfile.write(f"Error reading file: {file_path} (Unicode Decode Error)\n\n")
                except Exception as e:
                    outfile.write(f"Error reading file: {file_path} ({str(e)})\n\n")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process a directory and output its contents to a Markdown file.')
    parser.add_argument('-d', '--directory', type=str, help='Directory to process', default=os.getcwd())
    parser.add_argument('-o', '--output', type=str, help='Output Markdown file', default='directory_contents.md')
    parser.add_argument('-e', '--extensions', type=str, nargs='*', help='File extensions to include (e.g., py js)')

    # Parse arguments
    args = parser.parse_args()
    directory_to_process = args.directory
    output_markdown_file = args.output
    file_extensions = args.extensions

    # Debugging: Print starting process
    print(f"Processing directory: {directory_to_process}")
    print(f"File extensions: {file_extensions}")

    if not os.path.isdir(directory_to_process):
        print(f"Error: The directory '{directory_to_process}' does not exist.")
    else:
        process_directory(directory_to_process, output_markdown_file, file_extensions)
        print(f"Markdown file '{output_markdown_file}' has been created with the contents of '{directory_to_process}'.")

```

## directory_contents.md

```

```

## examples/test_augini.py

```
import pandas as pd
from augini import Augini
from augini.exceptions import APIError, DataProcessingError

def test_augini():
    # Initialize Augini
    api_key = "your_api_key"
    augini = Augini(api_key=api_key, use_openrouter=True, model='meta-llama/llama-3-8b-instruct', debug=False)

    # Create a sample DataFrame
    data = {
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Age': [30, 25, 45],
        'City': ['New York', 'Los Angeles', 'Chicago']
    }
    df = pd.DataFrame(data)

    # Test 1: Add a single feature
    try:
        result_df = augini.augment_single(df, 'Occupation')
    except (APIError, DataProcessingError) as e:
        print(f"Test 1 failed: {str(e)}")

    # Test 2: Add multiple features
    try:
        result_df = augini.augment_columns(df, 'Hobby', 'FavoriteColor')
    except (APIError, DataProcessingError) as e:
        print(f"Test 2 failed: {str(e)}")

    # Test 3: Add a feature with a custom prompt
    try:
        custom_prompt = "Based on the person's name and age, suggest a quirky pet for them. Respond with a JSON object with the key 'QuirkyPet'."
        result_df = augini.augment_single(df, 'QuirkyPet', custom_prompt=custom_prompt)
    except (APIError, DataProcessingError) as e:
        print(f"Test 3 failed: {str(e)}")

    # Test 4: Test error handling with an invalid API key
    try:
        invalid_augini = Augini(api_key="invalid_key", use_openrouter=True)
        invalid_augini.augment_single(df, 'InvalidFeature')
    except APIError:
        print("Test 4 passed: APIError caught as expected")

if __name__ == "__main__":
    test_augini()

```

## augini/__init__.py

```
from .core import Augini

__version__ = "0.1.0"
__all__ = ["Augini"]
```

## augini/core.py

```
import asyncio
from openai import AsyncOpenAI
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import nest_asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, ValidationError, root_validator
import re
from .utils import extract_json, generate_default_prompt

nest_asyncio.apply()

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

class CustomPromptModel(BaseModel):
    column_names: List[str]
    prompt: str
    available_columns: List[str]

    @root_validator(pre=True)
    def check_prompt(cls, values):
        prompt = values.get('prompt')
        available_columns = values.get('available_columns')
        placeholders = re.findall(r'{(.*?)}', prompt)
        for ph in placeholders:
            if ph not in available_columns:
                raise ValueError(f"Feature '{ph}' used in custom prompt does not exist in available columns: {available_columns}")
        return values


class Augini:
    def __init__(
        self,
        api_key: str,
        use_openrouter: bool = True,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.8,
        max_tokens: int = 500,
        concurrency_limit: int = 10,
        base_url: str = "https://openrouter.ai/api/v1",
        debug: bool = False
    ):
        self.client = AsyncOpenAI(
            base_url=base_url if use_openrouter else None,
            api_key=api_key
        )
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.debug = debug

        if debug:
            logger.setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)

    async def _get_response(self, prompt: str, row_data: Dict[str, Any], feature_names: List[str]) -> str:
        async with self.semaphore:
            try:
                if self.debug:
                    logger.debug(f"Prompt: {prompt}")
                
                json_template = "{" + ", ".join(f'"{name}": "FILL"' for name in feature_names) + "}"
                system_content = (
                    "You are a helpful and very creative assistant that generates hyperrealistic (but fictional) synthetic tabular data based on limited information. "
                    "Ensure the response is a valid JSON object as it is very important."
                )
                user_content = (
                    f"{prompt}\n\n"
                    f"Here is the row data: {row_data}\n\n"
                    f"Please fill the following JSON template with appropriate values:\n{json_template}"
                )

                if self.debug:
                    print(f"System content: {user_content}")
                    logger.debug(f"User content: {user_content}")


                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens, 
                    response_format={"type": "json_object"}
                )
                
                response_content = response.choices[0].message.content.strip()
                if self.debug:
                    logger.debug(f"Response: {response_content}")
                return response_content
            except Exception as e:
                logger.error(f"Error: {e}")
                raise APIError(f"API request failed: {str(e)}")

    async def _generate_features(self, df: pd.DataFrame, feature_names: List[str], prompt_template: str) -> pd.DataFrame:
        async def process_row(index: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
            try:
                row_data = row.to_dict()
                prompt = prompt_template.format(**row_data)
                logger.debug(f"Processing row {index}: {row_data}")
                response = await self._get_response(prompt, row_data, feature_names)
                logger.debug(f"Response for row {index}: {response}")
                feature_values = extract_json(response)
                logger.debug(f"Extracted features for row {index}: {feature_values}")
                if not feature_values or not all(feature in feature_values for feature in feature_names):
                    raise DataProcessingError(f"Expected features are missing in the JSON response: {feature_values}")
                return index, feature_values
            except Exception as e:
                logger.warning(f"Error processing row {index}: {e}")
                return index, {feature: np.nan for feature in feature_names}

        tasks = [process_row(index, row) for index, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
        
        # Sort results by index
        sorted_results = sorted(results, key=lambda x: x[0])
        
        for feature in feature_names:
            df[feature] = [result[1].get(feature, np.nan) for result in sorted_results]
        return df

    def _generate_features_sync(self, df: pd.DataFrame, feature_names: List[str], prompt_template: str) -> pd.DataFrame:
        results = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating features"):
            row_data = row.to_dict()
            prompt = prompt_template.format(**row_data)
            response = asyncio.run(self._get_response(prompt, row_data, feature_names))
            feature_values = extract_json(response)
            results.append(feature_values)
        
        for feature in feature_names:
            df[feature] = [result.get(feature, np.nan) for result in results]
        return df

    def augment_columns(self, df: pd.DataFrame, columns: List[str], custom_prompt: Optional[str] = None, use_sync: bool = False) -> pd.DataFrame:
        result_df = df.copy()
        available_columns = list(result_df.columns)
        column_names = columns

        if custom_prompt:
            try:
                CustomPromptModel(column_names=column_names, prompt=custom_prompt, available_columns=available_columns)
            except ValidationError as e:
                raise ValueError(f"Custom prompt validation error: {e}")

        prompt_template = custom_prompt or generate_default_prompt(column_names, available_columns)
        
        if use_sync:
            return self._generate_features_sync(result_df, column_names, prompt_template)
        else:
            return asyncio.run(self._generate_features(result_df, column_names, prompt_template))

    def augment_columns(self, df: pd.DataFrame, columns: List[str], custom_prompt: Optional[str] = None, use_sync: bool = False) -> pd.DataFrame:
        result_df = df.copy()
        available_columns = list(result_df.columns)
        column_names = columns

        if custom_prompt:
            try:
                CustomPromptModel(column_names=column_names, prompt=custom_prompt, available_columns=available_columns)
            except ValidationError as e:
                raise ValueError(f"Custom prompt validation error: {e}")
    

        prompt_template = custom_prompt or generate_default_prompt(column_names, available_columns)
        
        if use_sync:
            return self._generate_features_sync(result_df, column_names, prompt_template)
        else:
            return asyncio.run(self._generate_features(result_df, column_names, prompt_template))

class APIError(Exception):
    pass

class DataProcessingError(Exception):
    pass
```

## augini/utils.py

```
import json
import re

def extract_json(response):
    try:
        json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        return None

def generate_default_prompt(feature_names, available_columns):
    column_list = ", ".join(f"{col}: {{{col}}}" for col in available_columns)
    features = ", ".join(f'"{feature}": "<{feature}>"' for feature in feature_names)
    return (f"Given the following data:\n{column_list}\n"
            f"Please provide the following features in a JSON object:\n{features}\n"
            "If a feature is not applicable or cannot be determined, use null in the JSON.\n"
            "Ensure the response is a valid JSON object as it is very important.")



```

