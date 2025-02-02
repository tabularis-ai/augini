from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import os
from pathlib import Path
import yaml
import json


class AuginiConfig:
    """Configuration for Augini components."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 500,
        concurrency_limit: int = 5,
        debug: bool = False,
        log_level: str = "INFO",
    ):
        """Initialize configuration.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoint (optional)
            model: Model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            concurrency_limit: Maximum concurrent API calls
            debug: Enable debug mode
            log_level: Logging level
        """
        self.api_key = api_key
        self.base_url = base_url  # If None, OpenAI's default URL will be used
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.debug = debug
        self.log_level = log_level

    @classmethod
    def create_default(cls, api_key: str, **kwargs) -> "AuginiConfig":
        """Create configuration with default values."""
        return cls(api_key=api_key, **kwargs)

    # API Settings
    api_key: Optional[str] = None
    use_openrouter: bool = False
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "gpt-4-turbo-preview"

    # Model Settings
    temperature: float = 0.7
    max_tokens: int = 1000
    concurrency_limit: int = 5

    # Debug Settings
    debug: bool = False
    log_level: str = "INFO"

    # New parameter for markdown output in chat
    markdown_output: bool = True

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "api_key": "your-api-key",
                    "use_openrouter": True,
                    "model": "gpt-4o-mini",
                    "temperature": 0.8,
                    "debug": False,
                }
            ]
        },
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_TOKEN"),
            model=os.environ.get("AUGINI_MODEL", cls.__fields__["model"].default),
            temperature=float(
                os.environ.get("AUGINI_TEMPERATURE", cls.__fields__["temperature"].default)
            ),
            max_tokens=int(
                os.environ.get("AUGINI_MAX_TOKENS", cls.__fields__["max_tokens"].default)
            ),
            concurrency_limit=int(
                os.environ.get(
                    "AUGINI_CONCURRENCY_LIMIT", cls.__fields__["concurrency_limit"].default
                )
            ),
            debug=bool(os.environ.get("AUGINI_DEBUG", cls.__fields__["debug"].default)),
            log_level=os.environ.get("AUGINI_LOG_LEVEL", cls.__fields__["log_level"].default),
            markdown_output=bool(
                os.environ.get("AUGINI_MARKDOWN_OUTPUT", cls.__fields__["markdown_output"].default)
            ),
        )

    @classmethod
    def from_file(cls, file_path: str) -> "AuginiConfig":
        """Load configuration from a file (YAML or JSON)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path) as f:
            if path.suffix in [".yaml", ".yml"]:
                config_dict = yaml.safe_load(f)
            elif path.suffix == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError("Config file must be YAML or JSON")

        return cls(**config_dict)

    def to_file(self, file_path: str) -> None:
        """Save configuration to a file (YAML or JSON)."""
        path = Path(file_path)
        config_dict = self.model_dump()

        with open(file_path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif path.suffix == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError("Config file must be YAML or JSON")
