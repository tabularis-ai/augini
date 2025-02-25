"""
Augini - A LangChain-powered data analysis framework.
"""

from .config.base import AuginiConfig, ToolConfig, LLMConfig, AgentConfig, LLMProvider
from .agents.data_analyzer import DataAnalyzer
from .agents.data_chat import DataChat, ChatSession, ChatMessage
from .tools.base import AuginiBaseTool, ToolResult
from .tools.correlation import CorrelationAnalyzer
from .tools.missing_values import MissingValuesAnalyzer
from .tools.statistics import StatisticalSummary
from .tools.visualization import VisualizationTool

__version__ = "2.0.0"

__all__ = [
    "AuginiConfig",
    "ToolConfig",
    "LLMConfig",
    "AgentConfig",
    "LLMProvider",
    "DataAnalyzer",
    "AuginiBaseTool",
    "ToolResult",
    "CorrelationAnalyzer",
    "MissingValuesAnalyzer",
    "StatisticalSummary",
    "VisualizationTool",
    "DataChat",
    "ChatSession",
    "ChatMessage"
]
