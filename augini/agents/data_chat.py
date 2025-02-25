from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import uuid
import os
import tempfile
import matplotlib.pyplot as plt
import re

from .data_analyzer import DataAnalyzer, DataAnalyzerCallback
from ..config.base import AuginiConfig


class ChatMessage:
    """Represents a single message in the chat history."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """Initialize a chat message.
        
        Args:
            role: The role of the message sender (user or assistant)
            content: The content of the message
            timestamp: When the message was created (defaults to now)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create a message from a dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None
        message = cls(data["role"], data["content"], timestamp)
        if "id" in data:
            message.id = data["id"]
        return message


class ChatSession:
    """Represents a chat session with history tracking."""
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize a chat session.
        
        Args:
            session_id: Optional session identifier (defaults to a new UUID)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str) -> ChatMessage:
        """Add a message to the session.
        
        Args:
            role: The role of the message sender (user or assistant)
            content: The content of the message
            
        Returns:
            The created message
        """
        message = ChatMessage(role, content)
        self.messages.append(message)
        self.last_updated = datetime.now()
        return message
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the chat history as a list of dictionaries."""
        return [msg.to_dict() for msg in self.messages]
    
    def get_context_window(self, window_size: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages as context.
        
        Args:
            window_size: Number of recent messages to include
            
        Returns:
            List of recent messages as dictionaries
        """
        return [msg.to_dict() for msg in self.messages[-window_size:]]
    
    def clear(self) -> None:
        """Clear the chat history."""
        self.messages = []
        self.last_updated = datetime.now()


class DataChat:
    """Chat interface for interacting with the DataAnalyzer."""
    
    def __init__(self, config: AuginiConfig, analyzer: Optional[DataAnalyzer] = None):
        """Initialize the data chat interface.
        
        Args:
            config: Augini configuration
            analyzer: Optional DataAnalyzer instance (will create one if not provided)
        """
        self.config = config
        
        # Create or use provided DataAnalyzer
        self.analyzer = analyzer or DataAnalyzer(config)
        
        # Ensure the analyzer has an LLM initialized
        if not hasattr(self.analyzer, 'llm') or self.analyzer.llm is None:
            self.analyzer.llm = self.analyzer._initialize_llm()
            
        # Ensure the analyzer has a callback handler
        if not hasattr(self.analyzer, 'callback_handler') or self.analyzer.callback_handler is None:
            self.analyzer.callback_handler = DataAnalyzerCallback()
            
        self.session = ChatSession()
        self.current_df: Optional[pd.DataFrame] = None
    
    def ask(self, question: str, df: Optional[pd.DataFrame] = None) -> str:
        """Simplified method to ask a question about data.
        
        This is a convenience method that wraps the chat method.
        
        Args:
            question: The question to ask about the data
            df: Optional DataFrame to analyze (uses current_df if not provided)
            
        Returns:
            The answer to the question
        """
        return self.chat(question, df)
    
    def chat(self, message: str, df: Optional[pd.DataFrame] = None) -> str:
        """Process a user message and return a response.
        
        Args:
            message: The user's message
            df: Optional DataFrame to analyze (uses current_df if not provided)
            
        Returns:
            The assistant's response
        """
        if not message.strip():
            return "Please provide a question or command about your data."
        
        # Update the current DataFrame if provided
        if df is not None:
            self.current_df = df
        
        # Check if we have data to analyze
        if self.current_df is None:
            return "Please provide a DataFrame to analyze."
        
        # Track the user message
        self.track_interaction("user", message)
        
        # Process the query
        result = self.process_query(message)
        
        # Format the response
        response = self.format_response(result)
        
        # Track the assistant response
        self.track_interaction("assistant", response)
        
        return response
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a data analysis query.
        
        Args:
            query: The user's query about the data
            
        Returns:
            Dictionary containing analysis results
        """
        if self.current_df is None:
            return {
                "success": False,
                "error": "No DataFrame available for analysis",
                "error_type": "MissingDataError"
            }
        
        # Get chat history for context
        chat_history = [(msg.role, msg.content) for msg in self.session.messages[-10:]]
        
        try:
            # Use the DataAnalyzer to process the query
            result = self.analyzer.analyze(
                df=self.current_df,
                question=query,
                chat_history=chat_history
            )
            
            # Check if result is a dictionary with the expected structure
            if not isinstance(result, dict):
                return {
                    "success": False,
                    "question": query,
                    "error": "Analyzer returned an invalid result type",
                    "error_type": "InvalidResultError"
                }
            
            # Ensure the result has a success field
            if "success" not in result:
                result["success"] = True
                
            return result
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            print(f"Error in process_query: {str(e)}")
            print(error_traceback)
            
            return {
                "success": False,
                "question": query,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_traceback if self.config.debug else None
            }
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format analysis results into a user-friendly message using markdown.
        
        Args:
            result: Analysis results from the DataAnalyzer
            
        Returns:
            Formatted response message with markdown
        """
        if not result.get("success", False):
            error_msg = result.get("error", "An unknown error occurred")
            error_type = result.get("error_type", "Error")
            
            # Provide helpful suggestions based on error type
            if error_type == "MissingDataError":
                return f"**Error**: {error_msg}. Please provide a DataFrame to analyze."
            elif error_type == "ValueError":
                return f"**Error**: {error_msg}. Please check your query and try again."
            else:
                return f"**Sorry, I encountered an error**: {error_msg}. Please try rephrasing your question."
        
        # Get the raw answer
        answer = result.get("answer", "")
        raw_result = result.get("raw_result", {})
        
        # Format based on the type of analysis
        if "missing" in answer.lower():
            return self._format_missing_values(raw_result)
        elif "correlation" in answer.lower():
            return self._format_correlation(raw_result)
        elif "statistical" in answer.lower():
            return self._format_statistics(raw_result)
        elif "visualization" in answer.lower():
            return self._format_visualization(result)
        else:
            # Default formatting for other types of answers
            return self._format_default(answer, raw_result)
    
    def _format_missing_values(self, result: Dict[str, Any]) -> str:
        """Format missing values analysis results.
        
        Args:
            result: Missing values analysis results
            
        Returns:
            Formatted markdown string
        """
        if not result:
            return "I analyzed the missing values but couldn't find any specific patterns."
        
        missing_counts = result.get("missing_counts", {})
        total_missing = result.get("total_missing", 0)
        missing_percentage = result.get("missing_percentage", 0)
        
        response = [
            "## Missing Values Analysis",
            f"**Total missing values**: {total_missing} ({missing_percentage:.2f}% of all data points)",
            "\n### Missing Values by Column"
        ]
        
        if missing_counts:
            response.append("| Column | Missing Count | Missing % |")
            response.append("|--------|--------------|----------|")
            
            for col, count in missing_counts.items():
                percentage = count / len(self.current_df) * 100 if self.current_df is not None else 0
                response.append(f"| {col} | {count} | {percentage:.2f}% |")
        else:
            response.append("No missing values found in any column.")
        
        # Add recommendations if available
        if "recommendations" in result:
            response.append("\n### Recommendations")
            for rec in result.get("recommendations", []):
                response.append(f"- {rec}")
        
        return "\n".join(response)
    
    def _format_correlation(self, result: Dict[str, Any]) -> str:
        """Format correlation analysis results.
        
        Args:
            result: Correlation analysis results
            
        Returns:
            Formatted markdown string
        """
        if not result:
            return "I analyzed the correlations but couldn't find any significant relationships."
        
        correlation_matrix = result.get("correlation_matrix", {})
        strong_correlations = result.get("strong_correlations", [])
        
        response = ["## Correlation Analysis"]
        
        # Add summary of findings
        summary = result.get("summary", {})
        if summary:
            has_significant = summary.get("has_significant_correlations", False)
            if has_significant:
                response.append(f"**Found {summary.get('strong_correlations_found', 0)} strong correlations** among {summary.get('total_numeric_columns', 0)} numeric columns.")
            else:
                response.append("No strong correlations were found among the numeric columns.")
        
        # Add strong correlations if any
        if strong_correlations:
            response.append("\n### Strong Correlations")
            response.append("| Column 1 | Column 2 | Correlation |")
            response.append("|----------|----------|-------------|")
            
            for corr in strong_correlations:
                col1 = corr.get("column1", "")
                col2 = corr.get("column2", "")
                value = corr.get("value", 0)
                response.append(f"| {col1} | {col2} | {value:.3f} |")
        
        # Add correlation matrix (limited to a few columns for readability)
        if correlation_matrix:
            response.append("\n### Correlation Matrix")
            
            # Get column names
            columns = list(correlation_matrix.keys())
            if len(columns) > 5:  # Limit to 5 columns for readability
                columns = columns[:5]
                response.append("(Showing first 5 columns only)")
            
            # Create header row
            header = "| Column | " + " | ".join(columns) + " |"
            response.append(header)
            
            # Create separator row
            separator = "|--------|" + "|".join(["---------" for _ in columns]) + "|"
            response.append(separator)
            
            # Create data rows
            for col1 in columns:
                row = f"| {col1} |"
                for col2 in columns:
                    value = correlation_matrix.get(col1, {}).get(col2, 0)
                    row += f" {value:.3f} |"
                response.append(row)
        
        # Add interpretation
        response.append("\n### Interpretation")
        if strong_correlations:
            response.append("- **Strong positive correlation** (close to 1): As one variable increases, the other tends to increase.")
            response.append("- **Strong negative correlation** (close to -1): As one variable increases, the other tends to decrease.")
        else:
            response.append("- No strong correlations were found, suggesting the variables are largely independent of each other.")
        
        return "\n".join(response)
    
    def _format_statistics(self, result: Dict[str, Any]) -> str:
        """Format statistical analysis results.
        
        Args:
            result: Statistical analysis results
            
        Returns:
            Formatted markdown string
        """
        if not result:
            return "I analyzed the statistics but couldn't generate a meaningful summary."
        
        summary_stats = result.get("summary_stats", {})
        
        response = ["## Statistical Summary"]
        
        if summary_stats:
            # Create a table for each numeric column
            for col, stats in summary_stats.items():
                response.append(f"\n### {col}")
                response.append("| Statistic | Value |")
                response.append("|-----------|-------|")
                
                for stat, value in stats.items():
                    # Format the value based on its type
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:,.3f}" if isinstance(value, float) else f"{value:,}"
                    else:
                        formatted_value = str(value)
                    
                    response.append(f"| {stat.capitalize()} | {formatted_value} |")
        
        # Add distribution information if available
        distributions = result.get("distributions", {})
        if distributions:
            response.append("\n### Distribution Summary")
            for col, dist in distributions.items():
                response.append(f"- **{col}**: {dist}")
        
        return "\n".join(response)
    
    def _format_visualization(self, result: Dict[str, Any]) -> str:
        """Format visualization results.
        
        Args:
            result: Visualization results
            
        Returns:
            Formatted markdown string
        """
        if not result:
            return "I attempted to create a visualization but couldn't generate a meaningful result."
        
        if "error" in result:
            return f"**Visualization Error**: {result['error']}"
        
        plot_type = result.get("plot_type", "unknown")
        columns = result.get("columns", [])
        
        response = [
            "## Data Visualization",
            f"**Type**: {plot_type.capitalize()}"
        ]
        
        if columns:
            response.append(f"**Columns**: {', '.join(columns)}")
        
        # Add notes about the visualization
        response.append("\n### Notes")
        if plot_type == "histogram":
            response.append("- This histogram shows the distribution of values.")
            response.append("- The x-axis represents the values and the y-axis represents the frequency.")
        elif plot_type == "scatter":
            response.append("- This scatter plot shows the relationship between two variables.")
            response.append("- Each point represents an individual data point.")
        elif plot_type == "box":
            response.append("- This box plot shows the distribution of values.")
            response.append("- The box represents the interquartile range (IQR), with the median shown as a line.")
            response.append("- The whiskers extend to the min/max values within 1.5 * IQR.")
            response.append("- Points outside the whiskers are outliers.")
        
        # Add interpretation if available
        if "interpretation" in result:
            response.append("\n### Interpretation")
            response.append(result["interpretation"])
        
        # Add plot path if available (hidden in the message for retrieval)
        plot_path = None
        if isinstance(result, dict):
            plot_path = result.get("plot_path")
            if not plot_path and "visualization" in result:
                viz_result = result.get("visualization")
                if isinstance(viz_result, dict):
                    plot_path = viz_result.get("plot_path")
        
        if plot_path:
            # Add a hidden marker with the plot path for retrieval
            response.append(f"\n<!-- PLOT_PATH:{plot_path} -->")
            response.append("\n*A visualization has been generated and is available for display.*")
        else:
            response.append("\n*Note: The visualization would be displayed here in a graphical interface.*")
        
        return "\n".join(response)
    
    def _format_default(self, answer: str, result: Dict[str, Any]) -> str:
        """Format default results when no specific formatter is available.
        
        Args:
            answer: The raw answer string
            result: The raw result dictionary
            
        Returns:
            Formatted markdown string
        """
        # If the answer already looks formatted, return it as is
        if "##" in answer or "**" in answer:
            return answer
        
        # If we have DataFrame information
        if "shape" in result:
            shape = result.get("shape", (0, 0))
            columns = result.get("columns", [])
            
            response = [
                "## DataFrame Information",
                f"**Shape**: {shape[0]} rows × {shape[1]} columns",
                f"**Columns**: {', '.join(columns)}"
            ]
            
            # Add data types if available
            dtypes = result.get("dtypes", {})
            if dtypes:
                response.append("\n### Data Types")
                response.append("| Column | Type |")
                response.append("|--------|------|")
                
                for col, dtype in dtypes.items():
                    response.append(f"| {col} | {dtype} |")
            
            # Add sample data if available
            head = result.get("head", {})
            if head:
                response.append("\n### Sample Data (First 5 rows)")
                
                # Create header
                if columns:
                    response.append("| " + " | ".join(columns) + " |")
                    response.append("|" + "|".join(["-----" for _ in columns]) + "|")
                    
                    # Create rows
                    for i in range(min(5, shape[0])):
                        row_values = []
                        for col in columns:
                            col_data = head.get(col, {})
                            value = col_data.get(i, "") if isinstance(col_data, dict) else ""
                            row_values.append(str(value))
                        response.append("| " + " | ".join(row_values) + " |")
            
            return "\n".join(response)
        
        # Default formatting for any other type of answer
        return f"## Analysis Results\n\n{answer}"
    
    def track_interaction(self, role: str, content: str) -> None:
        """Store the current interaction in the session history.
        
        Args:
            role: The role of the message sender (user or assistant)
            content: The content of the message
        """
        self.session.add_message(role, content)
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Retrieve the current session's interactions.
        
        Returns:
            List of message dictionaries
        """
        return self.session.get_history()
    
    def clear_session(self) -> None:
        """Clear the current session history."""
        self.session.clear()
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set the current DataFrame for analysis.
        
        Args:
            df: DataFrame to analyze
        """
        self.current_df = df
    
    def handle_visualization_request(self, plot_type: str, **kwargs) -> Dict[str, Any]:
        """Handle a request to visualize data.
        
        Args:
            plot_type: Type of plot to generate
            **kwargs: Additional parameters for the visualization
            
        Returns:
            Dictionary containing visualization data
        """
        if self.current_df is None:
            return {
                "success": False,
                "error": "No DataFrame available for visualization",
                "error_type": "MissingDataError"
            }
        
        try:
            # Use the analyzer's visualization tool
            result = self.analyzer.analyze_visualizations(
                df=self.current_df,
                plot_type=plot_type,
                **kwargs
            )
            
            # Check if we have a plot path in the result
            if isinstance(result, dict) and "plot_path" not in result:
                # Generate a unique filename for the plot
                # Create a temporary directory if it doesn't exist
                plot_dir = os.path.join(tempfile.gettempdir(), "augini_plots")
                os.makedirs(plot_dir, exist_ok=True)
                
                # Generate a unique filename
                import uuid
                plot_path = os.path.join(plot_dir, f"plot_{uuid.uuid4()}.png")
                
                # Save the current figure if it exists
                if plt.get_fignums():
                    plt.savefig(plot_path)
                    plt.close()
                    
                    # Add the plot path to the result
                    if isinstance(result, dict):
                        result["plot_path"] = plot_path
                    else:
                        result = {"plot_path": plot_path, "original_result": result}
            
            return {
                "success": True,
                "visualization": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def get_last_plot_path(self) -> Optional[str]:
        """Get the path to the most recently generated plot.
        
        Returns:
            Path to the plot file or None if no plot was generated
        """
        # Check the last message for a plot path
        if not self.session.messages:
            return None
            
        last_message = self.session.messages[-1]
        if last_message.role != "assistant":
            return None
            
        # Look for plot path in the content
        match = re.search(r'PLOT_PATH:([^\s]+)', last_message.content)
        if match:
            return match.group(1)
            
        return None 