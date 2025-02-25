from typing import Optional, Dict, Any, ClassVar
import pandas as pd
import numpy as np
from scipy import stats
from langchain.callbacks.manager import CallbackManagerForToolRun

from .base import AuginiBaseTool, ToolResult
from ..config.base import ToolConfig


class StatisticalSummary(AuginiBaseTool):
    """Tool for generating comprehensive statistical summaries."""
    
    # Class variables for tool metadata
    name: ClassVar[str] = "statistical_summary"
    description: ClassVar[str] = """Generates detailed statistical summaries for DataFrame columns.
    Includes basic statistics, distribution analysis, and outlier detection."""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config=config)
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution of a numeric series."""
        # Remove NaN values for calculations
        clean_data = series.dropna()
        
        if len(clean_data) == 0:
            return {
                "error": "No non-null values in series"
            }
        
        # Basic statistics
        basic_stats = clean_data.describe().to_dict()
        
        # Additional statistics
        skewness = float(stats.skew(clean_data))
        kurtosis = float(stats.kurtosis(clean_data))
        
        # Detect outliers using IQR method
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
        
        # Test for normality
        _, normality_p_value = stats.normaltest(clean_data) if len(clean_data) >= 8 else (0, 0)
        
        return {
            "basic_stats": basic_stats,
            "distribution": {
                "skewness": round(skewness, 3),
                "kurtosis": round(kurtosis, 3),
                "is_normal": bool(normality_p_value > 0.05),
                "normality_p_value": round(float(normality_p_value), 3) if normality_p_value else None
            },
            "outliers": {
                "count": len(outliers),
                "percentage": round(len(outliers) / len(clean_data) * 100, 2),
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound)
                }
            },
            "missing_values": {
                "count": series.isna().sum(),
                "percentage": round(series.isna().mean() * 100, 2)
            }
        }
    
    def _analyze_categorical(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a categorical series."""
        # Remove NaN values for calculations
        clean_data = series.dropna()
        
        if len(clean_data) == 0:
            return {
                "error": "No non-null values in series"
            }
        
        # Value counts and frequencies
        value_counts = clean_data.value_counts()
        value_frequencies = clean_data.value_counts(normalize=True)
        
        return {
            "unique_values": {
                "count": len(value_counts),
                "values": list(value_counts.index)
            },
            "frequencies": {
                str(k): {
                    "count": int(value_counts[k]),
                    "percentage": round(float(value_frequencies[k]) * 100, 2)
                }
                for k in value_counts.index[:10]  # Limit to top 10 categories
            },
            "missing_values": {
                "count": series.isna().sum(),
                "percentage": round(series.isna().mean() * 100, 2)
            }
        }
    
    def _run(
        self,
        df: pd.DataFrame,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> ToolResult:
        """Generate statistical summaries for the DataFrame.
        
        Args:
            df: Input DataFrame to analyze
            run_manager: Callback manager for the tool run
            **kwargs: Additional keyword arguments
            
        Returns:
            ToolResult containing statistical summaries
        """
        try:
            # Select specific columns if provided
            columns = kwargs.get("columns")
            if columns is not None:
                df = df[columns]
            
            # Initialize results
            summaries = {}
            
            # Analyze each column
            for column in df.columns:
                series = df[column]
                
                if pd.api.types.is_numeric_dtype(series):
                    summaries[column] = {
                        "type": "numeric",
                        "analysis": self._analyze_distribution(series)
                    }
                else:
                    summaries[column] = {
                        "type": "categorical",
                        "analysis": self._analyze_categorical(series)
                    }
            
            # Generate overall summary
            total_rows = len(df)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            
            overall_summary = {
                "total_rows": total_rows,
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_columns),
                "categorical_columns": len(categorical_columns),
                "total_missing": df.isna().sum().sum(),
                "columns_with_missing": list(df.columns[df.isna().any()])
            }
            
            return ToolResult(
                success=True,
                result={
                    "overall_summary": overall_summary,
                    "column_summaries": summaries
                },
                metadata={
                    "columns_analyzed": list(df.columns),
                    "numeric_columns": list(numeric_columns),
                    "categorical_columns": list(categorical_columns),
                    "execution_time_seconds": 0  # TODO: Add actual timing
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                metadata={"exception_type": type(e).__name__}
            ) 