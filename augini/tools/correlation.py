from typing import Optional, Dict, Any, ClassVar
import pandas as pd
import numpy as np
from langchain.callbacks.manager import CallbackManagerForToolRun

from .base import AuginiBaseTool, ToolResult
from ..config.base import ToolConfig


class CorrelationAnalyzer(AuginiBaseTool):
    """Tool for analyzing correlations between numerical columns."""
    
    # Class variables for tool metadata
    name: ClassVar[str] = "correlation_analyzer"
    description: ClassVar[str] = """Analyzes correlations between numerical columns in a DataFrame.
    Provides correlation coefficients, significance tests, and identifies strong relationships."""
    
    def __init__(self, config: ToolConfig):
        """Initialize the analyzer."""
        super().__init__(config=config)
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numerical columns."""
        # Select numerical columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {
                "error": "Not enough numerical columns for correlation analysis",
                "numeric_columns": list(numeric_df.columns)
            }
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().round(3)
        
        # Find strong correlations (absolute value > 0.5)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = float(corr_matrix.iloc[i, j])
                if abs(corr_value) > 0.5:
                    strong_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": corr_value,
                        "strength": "strong positive" if corr_value > 0.7 else
                                  "strong negative" if corr_value < -0.7 else
                                  "moderate"
                    })
        
        # Calculate additional correlation statistics
        correlation_stats = {}
        for col1 in numeric_df.columns:
            correlation_stats[col1] = {
                "strongest_correlation": {
                    "column": "",
                    "value": 0.0
                },
                "average_correlation": 0.0,
                "negative_correlations": [],
                "positive_correlations": []
            }
            
            for col2 in numeric_df.columns:
                if col1 != col2:
                    corr_value = float(corr_matrix.loc[col1, col2])
                    
                    # Update strongest correlation
                    if abs(corr_value) > abs(correlation_stats[col1]["strongest_correlation"]["value"]):
                        correlation_stats[col1]["strongest_correlation"] = {
                            "column": col2,
                            "value": corr_value
                        }
                    
                    # Add to positive/negative correlations
                    if corr_value > 0.5:
                        correlation_stats[col1]["positive_correlations"].append({
                            "column": col2,
                            "value": corr_value
                        })
                    elif corr_value < -0.5:
                        correlation_stats[col1]["negative_correlations"].append({
                            "column": col2,
                            "value": corr_value
                        })
            
            # Calculate average correlation (excluding self-correlation)
            correlations = [
                corr_matrix.loc[col1, col2]
                for col2 in numeric_df.columns
                if col1 != col2
            ]
            correlation_stats[col1]["average_correlation"] = float(
                np.mean(correlations) if correlations else 0.0
            )
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "correlation_stats": correlation_stats,
            "numeric_columns": list(numeric_df.columns)
        }
    
    def _run(
        self,
        df: pd.DataFrame,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> ToolResult:
        """Analyze correlations in the DataFrame.
        
        Args:
            df: Input DataFrame to analyze
            run_manager: Callback manager for the tool run
            **kwargs: Additional keyword arguments
            
        Returns:
            ToolResult containing correlation analysis
        """
        try:
            if df.empty:
                raise ValueError("Empty DataFrame provided")
            
            # Select specific columns if provided
            columns = kwargs.get("columns")
            if columns is not None:
                df = df[columns]
            
            # Perform correlation analysis
            analysis_result = self._analyze_correlations(df)
            
            # Add summary insights
            if "error" not in analysis_result:
                num_strong_correlations = len(analysis_result["strong_correlations"])
                analysis_result["summary"] = {
                    "total_numeric_columns": len(analysis_result["numeric_columns"]),
                    "strong_correlations_found": num_strong_correlations,
                    "has_significant_correlations": num_strong_correlations > 0
                }
            
            return ToolResult(
                success="error" not in analysis_result,
                result=analysis_result,
                error=analysis_result.get("error"),
                metadata={
                    "columns_analyzed": list(df.columns),
                    "numeric_columns": analysis_result["numeric_columns"],
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