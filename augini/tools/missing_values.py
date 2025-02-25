from typing import Optional, Dict, Any, ClassVar
import pandas as pd
from langchain.callbacks.manager import CallbackManagerForToolRun

from .base import AuginiBaseTool, ToolResult
from ..config.base import ToolConfig


class MissingValuesAnalyzer(AuginiBaseTool):
    """Tool for analyzing missing values in a DataFrame."""
    
    # Class variables for tool metadata
    name: ClassVar[str] = "missing_values_analyzer"
    description: ClassVar[str] = """Analyzes missing values in a DataFrame.
    Provides counts and percentages of missing values for each column,
    as well as patterns in missingness."""
    
    def __init__(self, config: ToolConfig):
        """Initialize the analyzer."""
        super().__init__(config=config)
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing values."""
        # Get total number of rows
        total_rows = len(df)
        
        # Calculate missing value counts and percentages
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / total_rows * 100)
        
        # Find columns with any missing values
        columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
        
        # Analyze co-occurrence of missing values
        missing_patterns = {}
        if len(columns_with_missing) > 1:
            for col1 in columns_with_missing:
                missing_patterns[col1] = {}
                for col2 in columns_with_missing:
                    if col1 != col2:
                        # Count rows where both columns are missing
                        both_missing = df[df[col1].isnull() & df[col2].isnull()].shape[0]
                        # Calculate as percentage of rows where col1 is missing
                        if missing_counts[col1] > 0:
                            percentage = float((both_missing / missing_counts[col1] * 100))
                            missing_patterns[col1][col2] = percentage
        
        return {
            "total_rows": int(total_rows),
            "missing_counts": {col: int(count) for col, count in missing_counts.items()},
            "missing_percentages": {col: float(pct) for col, pct in missing_percentages.items()},
            "columns_with_missing": columns_with_missing,
            "missing_patterns": missing_patterns
        }
    
    def _run(
        self,
        df: pd.DataFrame,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> ToolResult:
        """Analyze missing values in the DataFrame.
        
        Args:
            df: Input DataFrame to analyze
            run_manager: Callback manager for the tool run
            **kwargs: Additional keyword arguments
            
        Returns:
            ToolResult containing missing value analysis
        """
        try:
            if df.empty:
                raise ValueError("Empty DataFrame provided")
            
            # Select specific columns if provided
            columns = kwargs.get("columns")
            if columns is not None:
                df = df[columns]
            
            # Perform missing value analysis
            analysis_result = self._analyze_missing_patterns(df)
            
            # Add summary insights
            total_missing = sum(analysis_result["missing_counts"].values())
            total_possible = analysis_result["total_rows"] * len(df.columns)
            overall_missing_percentage = float((total_missing / total_possible * 100))
            
            analysis_result["summary"] = {
                "total_missing_values": int(total_missing),
                "overall_missing_percentage": overall_missing_percentage,
                "columns_analyzed": len(df.columns)
            }
            
            return ToolResult(
                success=True,
                result=analysis_result,
                metadata={
                    "columns_analyzed": list(df.columns),
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