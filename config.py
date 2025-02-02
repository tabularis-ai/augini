"""Configuration management for Augini."""

from typing import Optional, Any
from pydantic import BaseModel, Field, model_validator

class AuginiConfig(BaseModel):
    """Configuration class for Augini."""
    
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field("gpt-4", description="Model to use for AI operations")
    temperature: float = Field(0.7, description="Temperature for AI operations")
    max_tokens: int = Field(2000, description="Maximum tokens for AI operations")
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    use_openrouter: bool = Field(False, description="Whether to use OpenRouter")
    memory_enabled: bool = Field(True, description="Whether to enable memory features")
    memory_k: int = Field(5, description="Number of memory items to keep")
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Model to use for embeddings")
    logging_level: str = Field("INFO", description="Logging level")

    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, values: Any) -> Any:
        """Validate configuration values."""
        if isinstance(values, dict):
            # Set default values if not provided
            values.setdefault('model', 'gpt-4')
            values.setdefault('temperature', 0.7)
            values.setdefault('max_tokens', 2000)
            values.setdefault('use_openrouter', False)
            values.setdefault('memory_enabled', True)
            values.setdefault('memory_k', 5)
            values.setdefault('embedding_model', 'all-MiniLM-L6-v2')
            values.setdefault('logging_level', 'INFO')
        return values

def create_default(api_key: str = None, **kwargs) -> AuginiConfig:
    """Create a default configuration instance.
    
    Args:
        api_key: OpenAI API key
        **kwargs: Additional configuration parameters to override defaults
    
    Returns:
        AuginiConfig: Configuration instance
    """
    config_dict = {
        "api_key": api_key,
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
        "base_url": None,
        "use_openrouter": False,
        "memory_enabled": True,
        "memory_k": 5,
        "embedding_model": "all-MiniLM-L6-v2",
        "logging_level": "INFO"
    }
    
    # Override defaults with provided kwargs
    config_dict.update(kwargs)
    
    return AuginiConfig(**config_dict)
