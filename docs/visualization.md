# Data Visualization with Augini

Augini provides powerful visualization capabilities to help you understand and communicate insights from your data. The `VisualizationTool` component supports multiple plotting backends and various visualization types, making it easy to create informative and attractive visualizations.

## Getting Started with Visualizations

Augini's visualization capabilities can be accessed in two ways:

1. Using the `VisualizationTool` directly
2. Using the `DataAnalyzer.analyze_visualizations()` method 

### Basic Example

```python
from augini import DataAnalyzer, AuginiConfig, ToolConfig

# Initialize with configuration
config = AuginiConfig(
    llm=dict(
        provider="openrouter",
        api_key="your-api-key",
        model="anthropic/claude-3.5-sonnet"
    ),
    tools={
        "visualization_tool": ToolConfig(
            name="visualization_tool",
            description="Generates visualizations from DataFrame data",
            enabled=True,
            parameters={
                "default_backend": "matplotlib",
                "default_style": "ggplot",
                "export_dir": "./visualizations"
            }
        )
    }
)

analyzer = DataAnalyzer(config)
df = your_dataframe  # Your pandas DataFrame

# Generate a scatter plot
result = analyzer.analyze_visualizations(
    df=df,
    plot_type="scatter",
    x="feature_1",
    y="feature_2",
    hue="category",
    title="Feature 1 vs Feature 2 by Category",
    export_format="png",
    export_filename="scatter_plot"
)

# The result contains:
# - The visualization data (base64-encoded image or JSON for Plotly)
# - The path to the exported file (if export_format and export_filename were provided)
# - The code to reproduce the visualization
```

## Supported Visualization Types

Augini supports a variety of visualization types across different backends:

### Common Plot Types (All Backends)

- **histogram**: Distribution of a single variable
- **scatter**: Relationship between two variables
- **line**: Trends over a continuous variable
- **bar**: Comparison across categories
- **box**: Distribution statistics across categories
- **heatmap**: Correlation matrix or 2D density

### Seaborn-Specific Plot Types

- **pairplot**: Matrix of scatter plots for multiple variables
- **violin**: Distribution across categories
- **kde**: Kernel density estimation plot

### Plotly-Specific Plot Types

- **bubble**: Scatter plot with size dimension
- **sunburst**: Hierarchical data visualization
- **treemap**: Hierarchical data as nested rectangles

## Visualization Backends

Augini supports multiple visualization backends:

### Matplotlib

The default backend, providing core plotting functionality with extensive customization options.

```python
result = analyzer.analyze_visualizations(
    df=df,
    plot_type="line",
    x="time",
    y="value",
    backend="matplotlib",
    style="seaborn-v0_8-darkgrid"
)
```

### Seaborn

Built on Matplotlib, Seaborn provides a high-level interface for creating attractive statistical graphics.

```python
result = analyzer.analyze_visualizations(
    df=df,
    plot_type="box",
    x="category",
    y="value",
    backend="seaborn",
    style="whitegrid"
)
```

### Plotly

An interactive visualization library that creates web-based visualizations.

```python
result = analyzer.analyze_visualizations(
    df=df,
    plot_type="scatter",
    x="feature_1",
    y="feature_2",
    hue="category",
    backend="plotly",
    export_format="html"
)
```

## Exporting Visualizations

Visualizations can be exported in various formats:

```python
result = analyzer.analyze_visualizations(
    df=df,
    plot_type="histogram",
    x="feature",
    export_format="png",  # Options: png, jpg, svg, pdf, html (Plotly only), json (Plotly only)
    export_filename="my_histogram"
)

# The visualization is saved to the export_dir specified in the configuration
# By default: ./visualizations/my_histogram.png
```

## Generating Visualization Code

One powerful feature of Augini's visualization capabilities is the automatic generation of code to reproduce the visualization. This makes it easy to customize and reuse visualizations outside of Augini.

```python
result = analyzer.analyze_visualizations(
    df=df,
    plot_type="scatter",
    x="feature_1",
    y="feature_2"
)

# The result contains the code to reproduce the visualization
print(result["result"]["code"])
```

Example generated code:

```python
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['feature_1'], df['feature_2'])
ax.set_xlabel('feature_1')
ax.set_ylabel('feature_2')
plt.tight_layout()
plt.show()
```

## Advanced Configuration

The `VisualizationTool` can be configured with various parameters:

```python
tool_config = ToolConfig(
    name="visualization_tool",
    description="Generates visualizations from DataFrame data",
    enabled=True,
    parameters={
        "default_backend": "matplotlib",  # Default plotting backend
        "default_style": "ggplot",        # Default visual style
        "default_figsize": (12, 8),       # Default figure size in inches
        "export_dir": "./my_visualizations"  # Directory for exported visualizations
    }
)
```

## Using the VisualizationTool Directly

For more control, you can use the `VisualizationTool` directly:

```python
from augini import VisualizationTool, ToolConfig

tool_config = ToolConfig(
    name="visualization_tool",
    description="Generates visualizations from DataFrame data",
    enabled=True
)

viz_tool = VisualizationTool(config=tool_config)

result = viz_tool._run(
    df=df,
    plot_type="scatter",
    x="feature_1",
    y="feature_2",
    hue="category",
    title="Feature Analysis",
    backend="seaborn",
    style="darkgrid",
    figsize=(12, 8),
    additional_params={"alpha": 0.7, "s": 100}
)
```

## Additional Parameters

Each plot type supports additional parameters that are passed directly to the underlying plotting function:

```python
# Histogram with custom bins and normalization
result = analyzer.analyze_visualizations(
    df=df,
    plot_type="histogram",
    x="feature",
    additional_params={
        "bins": 20,
        "density": True,
        "alpha": 0.7
    }
)

# Scatter plot with custom marker size and transparency
result = analyzer.analyze_visualizations(
    df=df,
    plot_type="scatter",
    x="feature_1",
    y="feature_2",
    additional_params={
        "s": 100,  # Marker size
        "alpha": 0.5,  # Transparency
        "edgecolor": "white"
    }
)
```

## Integration with Jupyter Notebooks

Visualizations can be easily displayed in Jupyter notebooks:

```python
from IPython.display import Image, display, HTML
import base64
from io import BytesIO

# For Matplotlib/Seaborn backends
result = analyzer.analyze_visualizations(df=df, plot_type="histogram", x="feature")
img_data = base64.b64decode(result["plot_data"])
display(Image(img_data))

# For Plotly backend
result = analyzer.analyze_visualizations(
    df=df, 
    plot_type="scatter", 
    x="feature_1", 
    y="feature_2",
    backend="plotly"
)
display(HTML(result["plot_data"]))
``` 