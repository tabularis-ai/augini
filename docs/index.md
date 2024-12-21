## *Build and Enhance Custom Datasets for your Use Case*

<p align="center">
  <img src="assets/images/logo_augini.png" alt="augini logo" width="200"/>
</p>

<a href="https://discord.gg/sznxwdqBXj">
  <img src="https://img.shields.io/badge/Discord-7289DA?&amp;logo=discord&amp;logoColor=white">
</a>


# AI-Powered Tabular Data Augmentation, Generation, Labeling, and Anonymization 

`augini` is a versatile Python framework that leverages AI for comprehensive data manipulation. It uses large language models to augment, generate, and anonymize tabular data, creating realistic and privacy-preserving datasets.


## Data Augmentation:

- Enhance existing datasets with AI-generated features
- Add contextual information based on current data
- Infuse domain knowledge from LLMs


## Synthetic Data Generation and Extantion:

- Create entirely new, realistic datasets
- Maintain statistical properties of original data
- Generate diverse, coherent synthetic profiles


## Data Anonymization:

- Implement k-anonymity and l-diversity
- Generate synthetic identifiers
- Balance privacy and data utility

## Use Cases

- Augment ML training datasets
- Generate privacy-safe data for sharing
- Automatic labeling using state-of-the-art AI models 
- Create synthetic data for software testing
- Develop realistic scenarios for business planning
- Produce diverse datasets for research and education


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

api_key = "OpenAI or OpenRouter token"

# OpenAI
augini = Augini(api_key=api_key,  model='gpt-4-turbo', use_openrouter=False)

# OpenRouter 
augini = Augini(api_key=api_key, use_openrouter=True, model='meta-llama/llama-3-8b-instruct')

# Create a sample DataFrame
data = {
    'Place of Birth': ['New York', 'London', 'Tokyo'],
    'Age': [30, 25, 40],
    'Gender': ['Male', 'Female', 'Male']
}
df = pd.DataFrame(data)

# Add synthetic features
result_df = augini.augment_columns(df, ['NAME', 'OCCUPATION', 'FAVORITE_DRINK'])

print(result_df)
```

### Contact us
- [info@tabularis.ai](mailto:info@tabularis.ai)

