"""Custom exceptions for the Augini framework."""

class AuginiError(Exception):
    """Base exception for all Augini errors."""
    pass


class ConfigurationError(AuginiError):
    """Raised when there is a configuration error."""
    pass


class ToolError(AuginiError):
    """Raised when there is an error in a tool's execution."""
    pass


class AgentError(AuginiError):
    """Raised when there is an error in agent execution."""
    pass


class ValidationError(AuginiError):
    """Raised when there is a validation error."""
    pass


class DataProcessingError(AuginiError):
    """Raised when there's an error processing data."""
    pass


class DataQualityError(DataProcessingError):
    """Raised when data doesn't meet quality standards."""
    pass
