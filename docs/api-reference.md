# API Reference

Complete reference for Augini's core APIs.

## DataEngineer

The `DataEngineer` class provides data transformation and feature engineering capabilities.

### Methods

#### transform()
```python
def transform(
    data: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame
```

Transform input data and generate new features.

**Parameters:**
- `data` (pd.DataFrame): Input dataset
- `target_columns` (List[str], optional): Columns to generate
- `**kwargs`: Additional transformation options

**Returns:**
- pd.DataFrame: Transformed dataset

## DataAnalyzer

The `DataAnalyzer` class provides data analysis and insight generation capabilities.

### Methods

#### analyze()
```python
def analyze(
    data: pd.DataFrame,
    analysis_type: str = 'statistical',
    **kwargs
) -> Dict[str, Any]
```

Analyze input data and generate insights.

**Parameters:**
- `data` (pd.DataFrame): Input dataset
- `analysis_type` (str): Type of analysis ('statistical', 'trends', etc.)
- `**kwargs`: Additional analysis options

**Returns:**
- Dict[str, Any]: Analysis results

## Configuration

The `AuginiConfig` class manages configuration settings.

### Methods

#### from_env()
```python
@classmethod
def from_env(cls) -> 'AuginiConfig'
```

Load configuration from environment variables.

#### from_file()
```python
@classmethod
def from_file(
    cls,
    file_path: str
) -> 'AuginiConfig'
```

Load configuration from a YAML/JSON file.

### Configuration Options

Key configuration parameters:

```yaml
# API Settings
api_key: str  # Your API key
model: str    # Model to use (default: gpt-4-turbo-preview)

# Processing Settings
batch_size: int     # Batch size for processing
temperature: float  # Model temperature

# Debug Settings
debug: bool      # Enable debug mode
log_level: str   # Logging level
``` 