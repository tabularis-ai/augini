from typing import List, Dict, Any, Optional
import pandas as pd
from langchain.agents import AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS

from ..config.base import AuginiConfig, ToolConfig
from ..tools.base import AuginiBaseTool
from ..tools.missing_values import MissingValuesAnalyzer
from ..tools.correlation import CorrelationAnalyzer
from ..tools.statistics import StatisticalSummary
from ..tools.visualization import VisualizationTool


# Custom prompt template for the agent
AGENT_PROMPT = """You are a data analysis expert using pandas and various analysis tools.
Your goal is to answer questions about the data by using the available tools effectively.

When analyzing data, follow these guidelines:
1. For questions about missing values, use the missing_values_analyzer tool
2. For questions about relationships between numerical columns, use the correlation_analyzer tool
3. For questions about basic statistics or distributions, use the statistical_summary tool
4. For visualization requests or when visual analysis would be helpful, use the visualization_tool
5. For complex questions, combine multiple tools to provide comprehensive answers

The DataFrame you are analyzing is stored in the 'df' variable.

{format_instructions}

Question: {input}
{agent_scratchpad}"""


class DataAnalyzerCallback(BaseCallbackHandler):
    """Callback handler for the DataAnalyzer agent."""
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> Any:
        """Called when agent takes an action."""
        print(f"\nAgent action: {action.tool}\nInput: {action.tool_input}\n")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> Any:
        """Called when agent finishes."""
        print(f"\nAgent finished: {finish.return_values}\n")


class DataAnalyzer:
    """Agent for analyzing data using LangChain and custom tools."""
    
    def __init__(self, config: AuginiConfig):
        """Initialize the data analyzer with configuration."""
        self.config = config
        self.df = None
        self.agent = None
        self.tools = self._initialize_tools()
        
        # Initialize the LLM based on configuration
        self.llm = self._initialize_llm()
        
        # Initialize callback handler
        self.callback_handler = DataAnalyzerCallback()
        
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        # Create LLM based on provider
        if self.config.llm.provider == "openai" or self.config.llm.provider == "openrouter":
            return ChatOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                max_tokens=self.config.llm.max_tokens,
                streaming=self.config.llm.streaming
            )
        elif self.config.llm.provider == "anthropic":
            # You would need to import the appropriate class for Anthropic
            # For now, we'll use ChatOpenAI as a placeholder
            return ChatOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                max_tokens=self.config.llm.max_tokens,
                streaming=self.config.llm.streaming
            )
        else:
            # Default to OpenAI
            return ChatOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                api_key=self.config.llm.api_key
            )
    
    def _initialize_tools(self) -> List[AuginiBaseTool]:
        """Initialize the analysis tools based on configuration."""
        tools = []
        
        # Add missing values analyzer if enabled
        missing_values_enabled = True
        if "missing_values_analyzer" in self.config.tools:
            tool_config = self.config.tools["missing_values_analyzer"]
            missing_values_enabled = getattr(tool_config, "enabled", True)
        
        if missing_values_enabled:
            # Create default config if not present
            if "missing_values_analyzer" not in self.config.tools:
                missing_values_config = ToolConfig(
                    name="missing_values_analyzer",
                    description="Analyzes missing values in the DataFrame",
                    enabled=True
                )
            else:
                missing_values_config = self.config.tools["missing_values_analyzer"]
            
            tools.append(MissingValuesAnalyzer(config=missing_values_config))
            
        # Add correlation analyzer if enabled
        correlation_enabled = True
        if "correlation_analyzer" in self.config.tools:
            tool_config = self.config.tools["correlation_analyzer"]
            correlation_enabled = getattr(tool_config, "enabled", True)
        
        if correlation_enabled:
            # Create default config if not present
            if "correlation_analyzer" not in self.config.tools:
                correlation_config = ToolConfig(
                    name="correlation_analyzer",
                    description="Analyzes correlations between columns",
                    enabled=True
                )
            else:
                correlation_config = self.config.tools["correlation_analyzer"]
            
            tools.append(CorrelationAnalyzer(config=correlation_config))
            
        # Add statistical summary if enabled
        stats_enabled = True
        if "statistical_summary" in self.config.tools:
            tool_config = self.config.tools["statistical_summary"]
            stats_enabled = getattr(tool_config, "enabled", True)
        
        if stats_enabled:
            # Create default config if not present
            if "statistical_summary" not in self.config.tools:
                stats_config = ToolConfig(
                    name="statistical_summary",
                    description="Provides statistical summaries of the data",
                    enabled=True
                )
            else:
                stats_config = self.config.tools["statistical_summary"]
            
            tools.append(StatisticalSummary(config=stats_config))
            
        # Add visualization tool if enabled
        viz_enabled = True
        if "visualization_tool" in self.config.tools:
            tool_config = self.config.tools["visualization_tool"]
            viz_enabled = getattr(tool_config, "enabled", True)
        
        if viz_enabled:
            # Create default config if not present
            if "visualization_tool" not in self.config.tools:
                viz_config = ToolConfig(
                    name="visualization_tool",
                    description="Generates visualizations from DataFrame data",
                    enabled=True
                )
            else:
                viz_config = self.config.tools["visualization_tool"]
            
            tools.append(VisualizationTool(config=viz_config))
            
        return tools
    
    def analyze_missing_values(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze missing values in the DataFrame.
        
        Args:
            df: Input DataFrame to analyze
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing missing value analysis results
        """
        tool = next(t for t in self.tools if isinstance(t, MissingValuesAnalyzer))
        return tool.analyze(df, **kwargs)
    
    def analyze_correlations(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze correlations in the DataFrame.
        
        Args:
            df: Input DataFrame to analyze
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing correlation analysis results
        """
        tool = next(t for t in self.tools if isinstance(t, CorrelationAnalyzer))
        return tool.analyze(df, **kwargs)
    
    def analyze_statistics(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze statistical properties of the DataFrame.
        
        Args:
            df: Input DataFrame to analyze
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing statistical analysis results
        """
        tool = next(t for t in self.tools if isinstance(t, StatisticalSummary))
        return tool.analyze(df, **kwargs)
    
    def analyze_visualizations(self, df: pd.DataFrame, plot_type: str, **kwargs) -> Dict[str, Any]:
        """Generate visualizations for the data.
        
        Args:
            df: DataFrame to visualize
            plot_type: Type of plot to generate
            **kwargs: Additional parameters for the visualization
            
        Returns:
            Dictionary containing visualization data and metadata
        """
        # Find the visualization tool
        viz_tool = next((tool for tool in self.tools if isinstance(tool, VisualizationTool)), None)
        
        if not viz_tool:
            return {"error": "Visualization tool not available"}
        
        # Run the visualization tool
        result = viz_tool._run(df=df, plot_type=plot_type, **kwargs)
        
        return result.data if result.success else {"error": result.error}
    
    def update_config(self, config: AuginiConfig) -> None:
        """Update the analyzer configuration.
        
        Args:
            config: New configuration to use
        """
        self.config = config
        
        # Reinitialize tools with new config
        self.tools = self._initialize_tools()
    
    def analyze(
        self,
        df: pd.DataFrame,
        question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze the DataFrame based on the question.
        
        Args:
            df: Input DataFrame to analyze
            question: Natural language question about the data
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing analysis results and insights
        """
        if not question:
            raise ValueError("Question cannot be empty")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        try:
            # For now, let's implement a simple approach that doesn't rely on the Python REPL
            # We'll directly use our tools based on keywords in the question
            
            # Store the DataFrame for tool access
            self.df = df
            
            # Simple keyword-based routing to appropriate tools
            result = None
            
            # Check for missing values related questions
            if any(keyword in question.lower() for keyword in ["missing", "null", "na", "nan"]):
                result = self.analyze_missing_values(df)
                answer = f"Missing values analysis: {result}"
            
            # Check for correlation related questions
            elif any(keyword in question.lower() for keyword in ["correlation", "relationship", "related", "connect"]):
                result = self.analyze_correlations(df)
                answer = f"Correlation analysis: {result}"
            
            # Check for statistics related questions
            elif any(keyword in question.lower() for keyword in ["statistics", "summary", "describe", "mean", "median", "std", "min", "max"]):
                result = self.analyze_statistics(df)
                answer = f"Statistical summary: {result}"
            
            # Check for visualization related questions
            elif any(keyword in question.lower() for keyword in ["plot", "chart", "graph", "visualize", "visualization", "show"]):
                # Default to histogram for now
                result = self.analyze_visualizations(df, plot_type="histogram")
                answer = f"Visualization created: {result}"
            
            # Default response if no specific tool matches
            else:
                # Basic DataFrame info as fallback
                info = {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "head": df.head(5).to_dict()
                }
                answer = f"DataFrame information: {info}"
            
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "raw_result": result
            }
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            print(f"Error in analyze: {str(e)}")
            print(error_traceback)
            
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_traceback if self.config.debug else None
            }
    
    async def aanalyze(
        self,
        df: pd.DataFrame,
        question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Async version of analyze()."""
        # TODO: Implement true async version
        return self.analyze(df, question, **kwargs) 