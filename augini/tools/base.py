from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, ClassVar
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
import json

from ..config.base import ToolConfig


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON serializable types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    return obj


class DataFrameInput(BaseModel):
    """Base input schema for DataFrame operations."""
    dataframe: Any = Field(..., description="The input pandas DataFrame")
    columns: Optional[List[str]] = Field(
        default=None,
        description="Specific columns to analyze. If None, all columns are used."
    )
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolResult(BaseModel):
    """Standardized output format for all tools."""
    success: bool = Field(..., description="Whether the operation was successful")
    result: Any = Field(..., description="The result of the operation")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the operation"
    )


class AuginiBaseTool(BaseTool):
    """Base class for all Augini tools."""
    
    # Class variables for tool metadata
    name: ClassVar[str]
    description: ClassVar[str]
    
    def __init__(self, config: ToolConfig, **kwargs):
        """Initialize the tool with configuration."""
        if not config.enabled:
            raise ValueError(f"Tool {config.name} is disabled in configuration")
        
        # Initialize base class with name and description from config
        super().__init__(
            name=config.name,
            description=config.description,
            **kwargs
        )
        
        # Store config
        self._config = config
    
    @property
    def config(self) -> ToolConfig:
        """Get the tool configuration."""
        return self._config
    
    def analyze(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze the DataFrame using this tool.
        
        Args:
            df: Input DataFrame to analyze
            columns: Specific columns to analyze. If None, all columns are used.
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing analysis results
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")
            
        # Create input dictionary
        input_dict = {
            "dataframe": df.to_dict('records'),  # Use records format for better serialization
            "columns": columns
        }
        
        # Run the tool
        result = json.loads(self.run(json.dumps(input_dict)))
        
        # Return the inner result if successful, otherwise raise the error
        if result.get("success", False):
            return result.get("result", {})
        else:
            raise ValueError(result.get("error", "Unknown error occurred"))
    
    @abstractmethod
    def _run(
        self,
        df: pd.DataFrame,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> ToolResult:
        """Core implementation of the tool's functionality.
        
        Args:
            df: Input DataFrame to analyze
            run_manager: Callback manager for the tool run
            **kwargs: Additional keyword arguments
            
        Returns:
            ToolResult containing the operation result
        """
        pass
    
    def _arun(
        self,
        df: pd.DataFrame,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> ToolResult:
        """Async implementation of the tool's functionality.
        
        By default, calls the sync implementation. Override for true async behavior.
        """
        return self._run(df, run_manager=run_manager, **kwargs)
    
    def run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Run the tool with the given input.
        
        Args:
            tool_input: JSON string containing input parameters
            run_manager: Callback manager for the tool run
            **kwargs: Additional keyword arguments
            
        Returns:
            JSON string containing the tool result
        """
        try:
            # Parse input JSON
            input_dict = json.loads(tool_input)
            
            # Convert DataFrame from dict representation
            if "dataframe" in input_dict:
                input_dict["dataframe"] = pd.DataFrame(input_dict["dataframe"])
            
            # Validate input
            validated_input = DataFrameInput(**input_dict)
            
            # Run the tool implementation
            result = self._run(
                df=validated_input.dataframe,
                columns=validated_input.columns,
                run_manager=run_manager,
                **kwargs
            )
            
            # Convert result to JSON-serializable format
            result_dict = result.model_dump()
            result_dict["result"] = convert_to_json_serializable(result_dict["result"])
            
            # Convert result to JSON string
            return json.dumps(result_dict)
            
        except Exception as e:
            error_result = ToolResult(
                success=False,
                result=None,
                error=str(e),
                metadata={"exception_type": type(e).__name__}
            )
            return json.dumps(error_result.model_dump())
    
    async def arun(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Async version of run()."""
        return self.run(tool_input, run_manager=run_manager, **kwargs) 