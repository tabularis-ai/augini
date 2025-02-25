from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
import os
from pathlib import Path
import yaml
import json
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


class PromptTemplate(BaseModel):
    """Configuration for prompt templates."""
    template: str
    input_variables: List[str]
    template_format: str = "f-string"
    validate_template: bool = True


class LLMConfig(BaseModel):
    """Configuration for LLM settings."""
    provider: LLMProvider = Field(
        default=LLMProvider.OPENROUTER,
        description="The LLM provider to use"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the LLM provider"
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for API endpoint"
    )
    model: str = Field(
        default="anthropic/claude-3-sonnet",
        description="Model to use for generation"
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature",
        ge=0.0,
        le=2.0
    )
    max_tokens: int = Field(
        default=1000,
        description="Maximum tokens in response",
        gt=0
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream responses"
    )


class AgentConfig(BaseModel):
    """Configuration for Augini agents."""
    max_iterations: int = Field(
        default=5,
        description="Maximum number of iterations for agent execution"
    )
    early_stopping: bool = Field(
        default=True,
        description="Whether to enable early stopping"
    )
    verbose: bool = Field(
        default=False,
        description="Whether to enable verbose output"
    )
    return_intermediate_steps: bool = Field(
        default=False,
        description="Whether to return intermediate steps in agent execution"
    )


class ToolConfig(BaseModel):
    """Configuration for individual tools."""
    name: str
    description: str
    enabled: bool = True
    required_packages: List[str] = Field(default_factory=list)
    timeout_seconds: int = Field(
        default=30,
        description="Maximum execution time in seconds"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific parameters"
    )


class AuginiConfig(BaseModel):
    """Enhanced configuration for Augini components with LangChain support."""
    
    # Core settings
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    
    # Tool configurations
    tools: Dict[str, ToolConfig] = Field(
        default_factory=dict,
        description="Configuration for individual tools"
    )
    
    # System settings
    concurrency_limit: int = Field(
        default=5,
        description="Maximum concurrent operations"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory for caching results"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "llm": {
                        "provider": "openrouter",
                        "api_key": "your-api-key",
                        "model": "anthropic/claude-3-sonnet",
                        "temperature": 0.7
                    },
                    "debug": True
                }
            ]
        }
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @classmethod
    def from_env(cls) -> "AuginiConfig":
        """Load configuration from environment variables."""
        llm_config = LLMConfig(
            provider=os.environ.get("AUGINI_PROVIDER", LLMProvider.OPENROUTER),
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_TOKEN"),
            model=os.environ.get("AUGINI_MODEL", "anthropic/claude-3-sonnet"),
            temperature=float(os.environ.get("AUGINI_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("AUGINI_MAX_TOKENS", "1000")),
            streaming=bool(os.environ.get("AUGINI_STREAMING", "false"))
        )
        
        return cls(
            llm=llm_config,
            concurrency_limit=int(os.environ.get("AUGINI_CONCURRENCY_LIMIT", "5")),
            debug=bool(os.environ.get("AUGINI_DEBUG", "false")),
            log_level=os.environ.get("AUGINI_LOG_LEVEL", "INFO"),
            cache_dir=os.environ.get("AUGINI_CACHE_DIR")
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

    def get_tool_config(self, tool_name: str) -> ToolConfig:
        """Get configuration for a specific tool."""
        if tool_name not in self.tools:
            # Return default configuration if tool not explicitly configured
            return ToolConfig(
                name=tool_name,
                description=f"Configuration for {tool_name}",
                enabled=True
            )
        return self.tools[tool_name] 