# Augini Documentation

<p align="center">
  <img src="assets/images/logo_augini.png" alt="augini logo" width="200"/>
</p>


## AI-Powered Tabular Data Framework

Augini is a Python framework that leverages AI for data manipulation and analysis through two powerful APIs:

### DataEngineer
Transform and prepare your data with automated:
- Feature engineering
- Data preprocessing
- Dataset scaling
- Data augmentation

### DataAnalyzer
Extract insights from your data using:
- Statistical analysis
- Trend detection
- Pattern recognition
- Visualization integration

## Quick Start

```python
from augini import DataEngineer, DataAnalyzer
import pandas as pd

# Sample customer data
data = pd.DataFrame({
    'age': [25, 35, 45, 28, 52],
    'income': [30000, 45000, 75000, 35000, 85000],
    'purchases': [150, 450, 850, 250, 950]
})

# Initialize with your API key
engineer = DataEngineer(api_key='your-api-key')

# Generate customer segments
data = engineer.generate_feature(
    df=data,
    name='customer_segment',
    description='Create customer segments based on age, income, and purchases',
    output_type='category'
)

# Initialize analyzer and fit the data
analyzer = DataAnalyzer(api_key='your-api-key')
analyzer.fit(data)

# Ask questions about the data
insights = analyzer.chat(
    "What are the characteristics of different customer segments? "
    "Focus on age, income, and purchase patterns."
)

print(insights)
```

## Documentation Sections

- [Quick Start & API Overview](quick-start.md) - Installation and basic usage
- [APIs](data-engineer.md) - Detailed API documentation
- [Chat Interface](chat.md) - Interactive data analysis
