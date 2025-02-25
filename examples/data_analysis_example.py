import pandas as pd
import numpy as np
from augini.config.base import AuginiConfig, ToolConfig
from augini.agents.data_analyzer import DataAnalyzer


def generate_sample_data(size: int = 1000) -> pd.DataFrame:
    """Generate sample data with missing values and correlations."""
    np.random.seed(42)
    
    # Generate correlated variables
    age = np.random.normal(35, 10, size)
    income = 20000 + 1000 * age + np.random.normal(0, 10000, size)  # Positive correlation with age
    education_years = 12 + 0.2 * age + np.random.normal(0, 2, size)  # Weak positive correlation with age
    satisfaction = 7 - 0.1 * age + np.random.normal(0, 2, size)  # Negative correlation with age
    
    # Generate categorical variables
    customer_types = np.random.choice(['Basic', 'Premium', 'Enterprise'], size)
    subscription_lengths = np.random.normal(24, 12, size).round()
    
    # Create DataFrame
    data = {
        'user_id': range(1, size + 1),
        'age': age.round(),
        'income': income.round(),
        'education_years': education_years.round(),
        'satisfaction_score': satisfaction.round(2),
        'customer_type': customer_types,
        'subscription_length': subscription_lengths.round()
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values with patterns
    # More missing values in income for older customers
    age_mask = df['age'] > 50
    df.loc[age_mask & (np.random.rand(len(df)) < 0.2), 'income'] = np.nan
    
    # More missing satisfaction scores for lower-income customers
    income_mask = df['income'] < 40000
    df.loc[income_mask & (np.random.rand(len(df)) < 0.15), 'satisfaction_score'] = np.nan
    
    # Random missing values in education years
    random_mask = np.random.rand(size) < 0.1
    df.loc[random_mask, 'education_years'] = np.nan
    
    return df


def main():
    # Generate sample data
    df = generate_sample_data()
    print("\nSample Data:")
    print("-" * 80)
    print(df.head())
    print("\nDataFrame Info:")
    print("-" * 80)
    print(df.info())
    print("-" * 80)
    
    # Initialize configuration with tool-specific settings
    config = AuginiConfig(
        llm=dict(
            provider="openrouter",
            api_key="your-api-key-here",  # Replace with your API key
            model="anthropic/claude-3-sonnet",
            temperature=0.2,
            streaming=True
        ),
        agent=dict(
            max_iterations=5,
            early_stopping=True,
            verbose=True,
            return_intermediate_steps=True
        ),
        tools={
            "missing_values_analyzer": ToolConfig(
                name="missing_values_analyzer",
                description="Analyzes patterns in missing values",
                enabled=True
            ),
            "correlation_analyzer": ToolConfig(
                name="correlation_analyzer",
                description="Analyzes correlations between numerical columns",
                enabled=True
            ),
            "statistical_summary": ToolConfig(
                name="statistical_summary",
                description="Generates statistical summaries of columns",
                enabled=True
            )
        },
        debug=True
    )
    
    # Initialize analyzer
    analyzer = DataAnalyzer(config)
    
    # Example questions to demonstrate different tools
    questions = [
        # Missing Values Analysis
        "What is the overall pattern of missing values in this dataset? Are there any columns that have more missing values than others?",
        "Is there any relationship between age and missing values in the income column?",
        
        # Correlation Analysis
        "What are the strongest correlations between numerical columns in the dataset?",
        "How does age correlate with income and education years?",
        "Is there a relationship between satisfaction score and other numerical variables?",
        
        # Statistical Summary
        "What are the key statistics for age and income? Are there any outliers?",
        "How is the customer type distributed? What's the most common type?",
        "What's the distribution of satisfaction scores? Is it normal?",
        
        # Complex Analysis
        "Can you provide a comprehensive analysis of the relationship between age, income, and education years, including their distributions and correlations?",
        "How do satisfaction scores vary across different customer types? Include both statistical summary and missing value patterns in your analysis."
    ]
    
    # Run analysis for each question
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 80)
        
        result = analyzer.analyze(df, question)
        
        if result["success"]:
            print("\nAnswer:")
            print(result["answer"])
            
            if result.get("intermediate_steps"):
                print("\nIntermediate Steps:")
                for step in result["intermediate_steps"]:
                    print(f"- {step}")
        else:
            print(f"\nError: {result['error']}")
        
        print("-" * 80)


if __name__ == "__main__":
    main() 