# Data Analysis with Augini

Augini provides powerful data analysis capabilities through its DataAnalyzer component. This component helps you understand your data, discover insights, and generate visualizations.

## DataAnalyzer

The `DataAnalyzer` is the analysis component that provides structured access to various analysis tools.

### Basic Usage

```python
from augini import DataAnalyzer, AuginiConfig

# Initialize with configuration
config = AuginiConfig(
    llm=dict(
        provider="openrouter",
        api_key="your-api-key",
        model="anthropic/claude-3.5-sonnet"
    ),
    tools={
        "missing_values_analyzer": {"enabled": True},
        "correlation_analyzer": {"enabled": True},
        "statistical_summary": {"enabled": True},
        "visualization_tool": {"enabled": True}
    }
)

analyzer = DataAnalyzer(config)
df = your_dataframe  # Your pandas DataFrame

# Analyze missing values
missing_values_result = analyzer.analyze_missing_values(df)

# Analyze correlations
correlation_result = analyzer.analyze_correlations(df)

# Analyze statistics
statistics_result = analyzer.analyze_statistics(df)

# Generate a visualization
visualization_result = analyzer.analyze_visualizations(
    df=df,
    plot_type="scatter",
    x="feature_1",
    y="feature_2",
    hue="category"
)
```

## Configuration

The analyzer uses the following configuration structure:

```python
config = AuginiConfig(
    llm=dict(
        provider="openrouter",  # LLM provider
        api_key="your-api-key",  # API key
        base_url="https://openrouter.ai/api/v1",  # Base URL (optional)
        model="anthropic/claude-3.5-sonnet"  # Model name
    ),
    tools={
        # Tool configurations
        "visualization_tool": {
            "enabled": True,
            "parameters": {
                "default_backend": "seaborn",
                "default_style": "whitegrid",
                "export_dir": "./visualizations"
            }
        }
    },
    debug=False  # Enable debug mode
)
```

## Key Features

- Statistical insights
- Pattern detection
- Trend analysis
- Visualization capabilities

## Analysis Types

### Statistical Analysis

```python
# Get statistical analysis
stats = analyzer.analyze_statistics(df)
```

### Correlation Analysis

```python
# Get correlation analysis
correlations = analyzer.analyze_correlations(df)
```

### Missing Values Analysis

```python
# Get missing values analysis
missing = analyzer.analyze_missing_values(df)
```

### Visualization

```python
# Generate a visualization
visualization = analyzer.analyze_visualizations(
    df=df,
    plot_type="scatter",
    x="feature_1",
    y="feature_2",
    hue="category"
)
```

## Advanced Usage

### Using Multiple Tools Together

```python
# First analyze statistics
stats = analyzer.analyze_statistics(df)

# Then visualize correlations
correlations = analyzer.analyze_correlations(df)
visualization = analyzer.analyze_visualizations(
    df=df,
    plot_type="heatmap",
    x=correlations["result"]["correlation_matrix"]
)
``` 