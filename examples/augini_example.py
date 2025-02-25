"""Example usage of the Augini framework for data analysis."""

import pandas as pd
import numpy as np
from augini import (
    DataAnalyzer,
    AuginiConfig,
    ToolConfig,
    MissingValuesAnalyzer,
    CorrelationAnalyzer,
    StatisticalSummary
)


def generate_sample_data(size: int = 1000) -> pd.DataFrame:
    """Generate sample data with missing values and correlations."""
    np.random.seed(42)
    
    # Generate correlated variables
    age = np.random.normal(35, 10, size)
    income = 20000 + 1000 * age + np.random.normal(0, 10000, size)  # Positive correlation with age
    satisfaction = 7 - 0.1 * age + np.random.normal(0, 2, size)  # Negative correlation with age
    
    # Generate categorical variables
    customer_types = np.random.choice(['Basic', 'Premium', 'Enterprise'], size)
    subscription_lengths = np.random.normal(24, 12, size).round()
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age.round(),
        'income': income.round(),
        'satisfaction': satisfaction.round(2),
        'customer_type': customer_types,
        'subscription_length': subscription_lengths.round()
    })
    
    # Introduce missing values with patterns
    # More missing values in income for older customers
    age_mask = df['age'] > 50
    df.loc[age_mask & (np.random.rand(len(df)) < 0.2), 'income'] = np.nan
    
    # More missing satisfaction scores for lower-income customers
    income_mask = df['income'] < 40000
    df.loc[income_mask & (np.random.rand(len(df)) < 0.15), 'satisfaction'] = np.nan
    
    return df


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration."""
    print("\n=== Example 1: Basic Usage ===")
    
    # Initialize with default configuration
    config = AuginiConfig(
        llm=dict(
            provider="openrouter",
            api_key="your-api-key",  # Replace with your API key
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-sonnet"
        )
    )
    
    analyzer = DataAnalyzer(config)
    df = generate_sample_data(size=100)
    
    # Simple question about missing values
    question = "How many missing values are there in each column?"
    result = analyzer.analyze(df, question)
    
    print("\nQuestion:", question)
    print("Answer:", result["answer"])


def example_2_direct_tool_usage():
    """Example 2: Using analysis tools directly."""
    print("\n=== Example 2: Direct Tool Usage ===")
    
    # Create tool configurations
    tool_config = ToolConfig(
        name="missing_values_analyzer",
        description="Analyzes missing values",
        enabled=True
    )
    
    # Initialize tools directly
    missing_analyzer = MissingValuesAnalyzer(config=tool_config)
    correlation_analyzer = CorrelationAnalyzer(config=tool_config)
    stats_analyzer = StatisticalSummary(config=tool_config)
    
    df = generate_sample_data(size=100)
    
    # Use tools directly
    print("\nMissing Values Analysis:")
    missing_result = missing_analyzer.analyze(df)
    print(f"Total missing values: {missing_result['summary']['total_missing_values']}")
    print(f"Missing percentages: {missing_result['missing_percentages']}")
    
    print("\nCorrelation Analysis:")
    corr_result = correlation_analyzer.analyze(df)
    print("Strong correlations found:", len(corr_result['strong_correlations']))
    
    print("\nStatistical Summary:")
    stats_result = stats_analyzer.analyze(df)
    print("Columns analyzed:", stats_result['overall_summary']['total_columns'])


def example_3_custom_configuration():
    """Example 3: Using custom configuration for specific analysis needs."""
    print("\n=== Example 3: Custom Configuration ===")
    
    # Create custom configuration
    config = AuginiConfig(
        llm=dict(
            provider="openrouter",
            api_key="your-api-key",  # Replace with your API key
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-sonnet",
            temperature=0.2
        ),
        agent=dict(
            max_iterations=5,
            early_stopping=True,
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
    
    analyzer = DataAnalyzer(config)
    df = generate_sample_data(size=100)
    
    # Complex analysis question
    question = """Analyze the relationship between age, income, and satisfaction scores. 
    Include information about missing values and correlations. 
    Also provide basic statistical summaries for these variables."""
    
    result = analyzer.analyze(df, question)
    
    print("\nQuestion:", question)
    print("Answer:", result["answer"])
    
    if result.get("intermediate_steps"):
        print("\nIntermediate Analysis Steps:")
        for step in result["intermediate_steps"]:
            print(f"- {step}")


def example_4_specific_analyses():
    """Example 4: Running specific analyses using DataAnalyzer methods."""
    print("\n=== Example 4: Specific Analyses ===")
    
    config = AuginiConfig(
        llm=dict(
            provider="openrouter",
            api_key="your-api-key",  # Replace with your API key
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-sonnet"
        )
    )
    
    analyzer = DataAnalyzer(config)
    df = generate_sample_data(size=100)
    
    # Run specific analyses
    print("\nMissing Values Analysis:")
    missing_result = analyzer.analyze_missing_values(df)
    print(f"Total missing values: {missing_result['summary']['total_missing_values']}")
    
    print("\nCorrelation Analysis:")
    corr_result = analyzer.analyze_correlations(df)
    print("Number of strong correlations:", len(corr_result['strong_correlations']))
    
    print("\nStatistical Analysis:")
    stats_result = analyzer.analyze_statistics(df)
    print("Number of columns analyzed:", stats_result['overall_summary']['total_columns'])


def main():
    """Run all examples."""
    example_1_basic_usage()
    example_2_direct_tool_usage()
    example_3_custom_configuration()
    example_4_specific_analyses()


if __name__ == "__main__":
    main() 