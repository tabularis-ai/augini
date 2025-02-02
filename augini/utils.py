import json
import re
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import logging


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text, handling various formats."""
    try:
        # Try direct JSON parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON-like structure in text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


@lru_cache(maxsize=128)
def generate_default_prompt(feature_names: Tuple[str, ...], source_columns: Tuple[str, ...]) -> str:
    """Generate a default prompt for feature generation."""
    source_cols_str = ", ".join(source_columns)
    target_cols_str = ", ".join(feature_names)
    return f"Based on {source_cols_str}, generate realistic values for {target_cols_str}."


@lru_cache(maxsize=1024)
def calculate_row_hash(row_data: str) -> str:
    """Calculate hash for row data for caching purposes."""
    return hashlib.md5(row_data.encode()).hexdigest()


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types."""
    result = df.copy()

    # Optimize numeric columns
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for col in result.select_dtypes(include=numerics).columns:
        col_type = result[col].dtype

        # Downcast integers
        if col_type.kind == "i":
            result[col] = pd.to_numeric(result[col], downcast="integer")
        # Downcast floats
        elif col_type.kind == "f":
            result[col] = pd.to_numeric(result[col], downcast="float")

    # Optimize string columns
    for col in result.select_dtypes(include=["object"]).columns:
        num_unique = len(result[col].unique())
        num_total = len(result[col])

        # Convert to categorical if beneficial
        if num_unique / num_total < 0.5:  # Less than 50% unique values
            result[col] = result[col].astype("category")

    return result


def batch_dataframe(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    """Split DataFrame into batches for efficient processing."""
    return [df[i : i + batch_size] for i in range(0, len(df), batch_size)]


@lru_cache(maxsize=128)
def get_column_stats(df_hash: str, column: str) -> Dict[str, Any]:
    """Get cached statistics for a DataFrame column."""
    # Note: df_hash is used as part of the cache key
    return {
        "mean": df[column].mean() if pd.api.types.is_numeric_dtype(df[column]) else None,
        "unique_count": len(df[column].unique()),
        "null_count": df[column].isnull().sum(),
        "dtype": str(df[column].dtype),
    }


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate DataFrame for processing requirements."""
    issues = []

    # Check for null values
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        issues.append(f"Null values found in columns: {null_cols}")

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]
    if inf_cols:
        issues.append(f"Infinite values found in columns: {inf_cols}")

    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        issues.append("Duplicate column names found")

    return len(issues) == 0, issues


def configure_logging(debug: bool = False, log_level: str = "INFO"):
    """Configure logging settings for Augini.
    
    Args:
        debug: If True, sets more verbose logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set levels
    if debug:
        level = logging.DEBUG
    else:
        level = getattr(logging, log_level.upper())
    
    root_logger.setLevel(level)
    
    # Configure specific loggers
    loggers = ['openai', 'httpx', 'httpcore']
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
