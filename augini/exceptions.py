class AuginiError(Exception):
    """Base exception class for all Augini errors."""

    pass


class APIError(AuginiError):
    """Raised when there's an error with API communication."""

    pass


class AuthenticationError(APIError):
    """Raised when there's an authentication error with the API."""

    pass


class DataProcessingError(AuginiError):
    """Raised when there's an error processing data."""

    pass


class ValidationError(AuginiError):
    """Raised when there's an error validating input data or parameters."""

    pass


class ConfigurationError(AuginiError):
    """Raised when there's an error in configuration."""

    pass


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""

    pass


class NetworkError(APIError):
    """Raised when there are network communication issues."""

    pass


class DataQualityError(DataProcessingError):
    """Raised when generated data doesn't meet quality standards."""

    pass
