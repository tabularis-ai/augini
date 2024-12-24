[![PyPI version](https://badge.fury.io/py/augini.svg)](https://badge.fury.io/py/augini) [![Downloads](https://static.pepy.tech/badge/augini)](https://pepy.tech/project/augini)


## *Build and Enhance Custom Datasets for your Use Case*

<p align="center">
  <img src="img/logo_augini.png" alt="augini logo" width="200"/>
</p>


# augini: AI-Powered Tabular Data Augmentation, Generation, Labeling, and Anonymization 

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

api_key = "OpenAI or OpenRouter"

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


### Custom Prompts for Feature Generation

Custom prompts allow you to generate specific features based on your needs:

```python
custom_prompt = "Based on the person's name and age, suggest a quirky pet for them. Respond with a JSON object with the key 'QuirkyPet'."
result_df = augini.augment_single(df, 'QuirkyPet', custom_prompt=custom_prompt)
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

### Anonymizing Data

You can anonymize sensitive information in your dataset by generating synthetic data:


```python
from augini import Augini
import pandas as pd

api_key = "OpenAI or OpenRouter"

# OpenAI
augini = Augini(api_key=api_key, debug=False, use_openrouter=False, model='gpt-4-turbo')

# OpenRouter 
augini = Augini(api_key=api_key, use_openrouter=True, model='meta-llama/llama-3-8b-instruct')

# Create a sample DataFrame with sensitive information
data = {
    'Name': ['Alice Johnson', 'Bob Smith', 'Charlie Davis'],
    'Age': [28, 34, 45],
    'City': ['New York', 'Los Angeles', 'Chicago'],
    'Email': ['alice.johnson@example.com', 'bob.smith@example.com', 'charlie.davis@example.com'],
    'Phone': ['123-456-7890', '987-654-3210', '555-555-5555']
}
df = pd.DataFrame(data)

# Define a general anonymization prompt
anonymize_prompt = (
    "Given the information from the dataset, create an anonymized version that protects individual privacy while maintaining data utility. "
    "Follow these guidelines:\n\n"
    "1. K-Anonymity: Ensure that each combination of quasi-identifiers (e.g., age, city) appears at least k times in the dataset. "
    "Use generalization or suppression techniques as needed.\n"
    "2. L-Diversity: For sensitive attributes, ensure there are at least l well-represented values within each equivalence class.\n"
    "3. Direct Identifiers: Replace the following with synthetic data:\n"
    "   - Names: Generate culturally appropriate fictional names\n"
    "   - Email addresses: Create plausible fictional email addresses\n"
    "   - Phone numbers: Generate realistic but non-functional phone numbers\n"
    "4. Quasi-Identifiers: Apply generalization or suppression as needed:\n"
    "   - Age: Consider using age ranges instead of exact ages\n"
    "   - City: Use broader geographic regions if necessary\n"
    "5. Sensitive Attributes: Maintain the statistical distribution of sensitive data while ensuring diversity.\n"
    "6. Data Consistency: Ensure that the anonymized data remains internally consistent and plausible.\n"
    "7. Non-Sensitive Data: Keep unchanged unless required for k-anonymity or l-diversity.\n\n"
    "Respond with a JSON object containing the anonymized values for all fields. "
    "Ensure the anonymized dataset maintains utility for analysis while protecting individual privacy."
)

# Use the augment_columns method to anonymize the data
result_df = augini.augment_columns(df, ['Name_A', 'Email_A', 'Age_A', 'City_A'], custom_prompt=anonymize_prompt)

# Display the resulting DataFrame
print(result_df)
```
Output: 
```
            Name  Age         City                      Email         Phone            Name_A                       Email_A  Age_A      City_A
0  Alice Johnson   28     New York  alice.johnson@example.com  123-456-7890  Sophia Rodriguez  sophia.rodriguez@example.com  25-30  East Coast
1      Bob Smith   34  Los Angeles      bob.smith@example.com  987-654-3210     Sarah Johnson     sarah.johnson@example.org  30-39  West Coast
2  Charlie Davis   45      Chicago  charlie.davis@example.com  555-555-5555     Emily Johnson     emily.johnson@example.com  40-50     Midwest
```


### Automated Data Labeling

Augini can be used to automatically generate labels for data, enhancing datasets with semantic information. In this example, we use Augini to analyze sentences and generate semantic labels, sentiment analysis, and topic identification:


```python
from augini import Augini
import pandas as pd

# Initialize Augini
api_key = "your_api_key_here"
augini = Augini(api_key=api_key, use_openrouter=True, model='gpt-4-turbo')

# Create a sample DataFrame with sentences
data = {
    'sentence': [
        "The cat sat on the mat.",
        "I love to eat pizza on Fridays.",
        "The stock market crashed yesterday.",
        "She sang beautifully at the concert.",
        "The new policy will be implemented next month."
    ]
}
df = pd.DataFrame(data)

# Define custom prompts for labeling
semantic_label_prompt = """
Analyze the given sentence and provide a semantic label. Choose from the following options:
Statement
Opinion
Fact
Action
Event
Respond with a JSON object containing the key 'semantic_label' and its value.
"""

sentiment_prompt = """
Determine the sentiment of the given sentence. Choose from the following options:
Positive
Negative
Neutral
Respond with a JSON object containing the key 'sentiment' and its value.
"""

topic_prompt = """
Identify the main topic of the given sentence. Provide a short (1-3 words) topic label.
Respond with a JSON object containing the key 'topic' and its value.
"""

# Generate labels using Augini
result_df = augini.augment_columns(df, 
    ['semantic_label', 'sentiment', 'topic'],
    custom_prompt=f"Sentence: {{sentence}}\n\n{semantic_label_prompt}\n\n{sentiment_prompt}\n\n{topic_prompt}"
)

# Display the results
print(result_df)

# You can also save the results to a CSV file
result_df.to_csv('labeled_sentences.csv', index=False)
```

Output:

```
|    | sentence                                       | semantic_label   | sentiment   | topic   |
|---:|:-----------------------------------------------|:-----------------|:------------|:--------|
|  0 | The cat sat on the mat.                        | Statement        | Neutral     | Animal  |
|  1 | I love to eat pizza on Fridays.                | Opinion          | Positive    | Food    |
|  2 | The stock market crashed yesterday.            | Event            | Negative    | Finance |
|  3 | She sang beautifully at the concert.           | Statement        | Positive    | Music   |
|  4 | The new policy will be implemented next month. | Statement        | Neutral     | Policy  |
```

### Chat

`augini` allows you to gain more information about the intricacies of your data using the chat method. It enables interactive querying of pandas DataFrames using natural language while maintaining conversation history and uses contextual awareness to provide more relevant responses over time:

```python

# Create a sample customer dataset
np.random.seed(42)
n_customers = 100

data = {
    'CustomerID': [f'C{i:04d}' for i in range(1, n_customers + 1)],
    'Age': np.random.randint(18, 80, n_customers),
    'Tenure': np.random.randint(0, 10, n_customers),
    'MonthlyCharges': np.random.uniform(20, 200, n_customers).round(2),
    'TotalCharges': np.random.uniform(100, 5000, n_customers).round(2),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
    'Churn': np.random.choice([0, 1], n_customers, p=[0.7, 0.3])  # 30% churn rate
}

df = pd.DataFrame(data)

# We augment the data a bit.
augment_prompt = """
Based on the customer's age, tenure, monthly charges, total charges, contract type, and payment method, suggest:
1. A likely reason for churn (if applicable)
2. A personalized retention offer
3. The customer's estimated lifetime value (in dollars)

Respond with a JSON object with keys 'ChurnReason', 'RetentionOffer', and 'EstimatedLTV'.
"""

df = augini.augment_columns(df, ['ChurnReason', 'RetentionOffer', 'EstimatedLTV'], custom_prompt=augment_prompt)

# Ask a question about the data
response = augini.chat("What is the average of the LTV?", df)
print(response)

# Ask a followup question about the data
response = augini.chat("Can you tell me more about the mean as a metric?",df)
print(response)

# Get conversation history
full_history = augini.get_conversation_history(mode='full')
summary_history = augini.get_conversation_history(mode='summary') # This is used to get the summary conversation history instead(which is used for contextual awareness)

# Clear conversation history
augini.clear_conversation_history(mode='all')
```

### Contact 

If you are looking for an enterprise version of the tool or a need local version please contact us:

`info@tabularis.ai` 
