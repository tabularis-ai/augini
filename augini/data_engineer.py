import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import nest_asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field, ValidationError, model_validator
import json
import re
from openai import AsyncOpenAI
from .config import AuginiConfig
from .utils import extract_json, generate_default_prompt, configure_logging
from .exceptions import APIError, DataProcessingError

nest_asyncio.apply()

logger = logging.getLogger(__name__)

class FeatureSpec(BaseModel):
    """Feature generation specification."""

    name: str = Field(..., description="Name of the feature to generate")
    description: str = Field(..., description="Description of what to generate")
    output_type: str = Field(..., description="Type of output (float, category, text)")
    constraints: Optional[Dict] = Field(default=None, description="Value constraints")
    source_columns: Optional[List[str]] = Field(default=None, description="Input columns")


class DataEngineer:
    """AI-powered feature engineering assistant."""

    def __init__(
        self,
        api_key: str = None,
        config: Optional[AuginiConfig] = None,
        **kwargs
    ):
        """Initialize the engineer.

        Args:
            api_key: OpenAI/OpenRouter API key (optional if config is provided)
            config: AuginiConfig instance (optional)
            **kwargs: Override config values
        """
        # Initialize configuration
        if config is None:
            if api_key is None:
                raise ValueError("Either api_key or config must be provided")
            config = AuginiConfig.create_default(api_key=api_key)
        
        # Override config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        self.config = config
        
        # Initialize API client
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url  # If base_url is None, it will use OpenAI's default URL
        )

        # Configure logging
        if not self.config.debug:
            # Disable httpx logging
            logging.getLogger("httpx").setLevel(logging.WARNING)
            # Disable our own debug logging
            logging.getLogger("augini").setLevel(logging.WARNING)

        # Initialize semaphore for concurrent API calls
        self._semaphore = asyncio.Semaphore(self.config.concurrency_limit)

    async def _get_response(
        self, prompt: str, row_data: Dict[str, Any], feature_names: List[str]
    ) -> str:
        """Get response from OpenAI API.

        Args:
            prompt: Prompt to send to API
            row_data: Data for current row
            feature_names: Names of features to generate

        Returns:
            API response text
        """
        async with self._semaphore:
            try:
                if self.config.debug:
                    logger.debug(f"Prompt: {prompt}")

                system_content = (
                    "You are an expert data scientist specialized in feature engineering. "
                    "Your response must be a valid, complete JSON object with values for each requested feature. "
                    "Make sure to properly close all JSON brackets and escape special characters."
                )

                json_template = "{" + ", ".join(f'"{name}": "FILL"' for name in feature_names) + "}"

                user_content = (
                    f"{prompt}\n\n"
                    f"Here is the row data: {row_data}\n\n"
                    f"Please fill the following JSON template with appropriate values:\n{json_template}"
                )

                if self.config.debug:
                    print(f"System content: {system_content}")
                    logger.debug(f"User content: {user_content}")

                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )

                response_content = response.choices[0].message.content.strip()
                if self.config.debug:
                    logger.debug(f"Response: {response_content}")
                return response_content
            except Exception as e:
                logger.error(f"Error: {e}")
                raise APIError(f"API request failed: {str(e)}")

    async def _generate_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        prompt_template: str,
        progress_bar: Optional[tqdm] = None,
    ) -> pd.DataFrame:
        """Generate features for all rows asynchronously."""
        result_df = df.copy()  # Create a copy of original DataFrame
        tasks = []
        
        for idx, row in df.iterrows():
            task = asyncio.create_task(
                self._get_response(prompt_template, row.to_dict(), feature_names)
            )
            tasks.append((idx, task))

        for idx, task in tasks:
            try:
                response = await task
                feature_values = json.loads(response)
                
                # Add each feature value to the result DataFrame
                for feature_name in feature_names:
                    result_df.loc[idx, feature_name] = feature_values[feature_name]
                    
                if progress_bar:
                    progress_bar.update(1)
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                # Set default/null values for failed rows
                for feature_name in feature_names:
                    result_df.loc[idx, feature_name] = None

        return result_df

    def _generate_features_sync(
        self, df: pd.DataFrame, feature_names: List[str], prompt_template: str
    ) -> pd.DataFrame:
        """Generate features synchronously (for debugging/testing).

        Args:
            df: Input DataFrame
            feature_names: Names of features to generate
            prompt_template: Template for generation prompt

        Returns:
            DataFrame with generated features
        """
        return asyncio.run(self._generate_features(df, feature_names, prompt_template))

    def generate_feature(
        self,
        df: pd.DataFrame,
        name: str,
        description: str,
        output_type: str = "float",
        constraints: Optional[Dict] = None,
        source_columns: Optional[List[str]] = None,
        use_sync: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Generate a single feature.

        Args:
            df: Input DataFrame
            name: Name of feature to generate
            description: Description of what to generate
            output_type: Type of output (float, category, text)
            constraints: Value constraints
            source_columns: Input columns to use
            use_sync: Use synchronous processing
            show_progress: Show progress bar

        Returns:
            DataFrame with original columns plus new feature
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("DataFrame must not be empty")

        if not name:
            raise ValueError("name cannot be empty")
        
        if not description:
            raise ValueError("description cannot be empty")
        
        valid_types = ["float", "category", "text"]
        if output_type not in valid_types:
            raise ValueError("Unsupported output type")

        spec = FeatureSpec(
            name=name,
            description=description,
            output_type=output_type,
            constraints=constraints,
            source_columns=source_columns
        )

        prompt = self._create_feature_prompt(df, spec)

        if use_sync:
            result_df = self._generate_features_sync(df, [name], prompt)
        else:
            progress_bar = tqdm(total=len(df)) if show_progress else None
            result_df = asyncio.run(
                self._generate_features(df, [name], prompt, progress_bar)
            )
            if progress_bar:
                progress_bar.close()

        return result_df

    def _create_feature_prompt(self, df: pd.DataFrame, spec: FeatureSpec) -> str:
        """Create prompt for feature generation.

        Args:
            df: Input DataFrame
            spec: Feature specification

        Returns:
            Formatted prompt string
        """
        prompt = [
            f"Generate a {spec.output_type} feature named '{spec.name}'",
            f"Description: {spec.description}",
            "\nDataset Information:",
            f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns",
            f"- Columns: {', '.join(df.columns)}",
            f"- Data Types: {df.dtypes.to_dict()}\n",
        ]

        if spec.constraints:
            prompt.append(f"Constraints: {spec.constraints}\n")

        if spec.source_columns:
            prompt.append(f"Use these columns: {', '.join(spec.source_columns)}\n")

        return "\n".join(prompt)

    def generate_features(
        self,
        df: pd.DataFrame,
        features: List[Dict[str, Any]],
        use_sync: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Generate multiple features.

        Args:
            df: Input DataFrame
            features: List of feature specifications
            use_sync: Use synchronous processing
            show_progress: Show progress bar

        Returns:
            DataFrame with generated features
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame must not be empty")

        if not features:
            raise ValueError("Feature list cannot be empty")

        specs = []
        feature_names = []
        for feature in features:
            try:
                spec = FeatureSpec(**feature)
                specs.append(spec)
                feature_names.append(spec.name)
            except ValidationError as e:
                raise ValueError(f"Invalid feature specification: {e}")

        prompt = self._create_multi_feature_prompt(df, specs)

        if use_sync:
            result_df = self._generate_features_sync(df, feature_names, prompt)
        else:
            progress_bar = tqdm(total=len(df)) if show_progress else None
            result_df = asyncio.run(
                self._generate_features(df, feature_names, prompt, progress_bar)
            )
            if progress_bar:
                progress_bar.close()

        return result_df

    def _create_multi_feature_prompt(self, df: pd.DataFrame, specs: List[FeatureSpec]) -> str:
        """Create prompt for multiple feature generation.

        Args:
            df: Input DataFrame
            specs: List of feature specifications

        Returns:
            Formatted prompt string
        """
        prompt = [
            "Generate multiple features with these specifications:",
            "\nFeatures to generate:"
        ]

        for spec in specs:
            prompt.extend([
                f"\n{spec.name}:",
                f"- Type: {spec.output_type}",
                f"- Description: {spec.description}"
            ])
            if spec.constraints:
                prompt.append(f"- Constraints: {spec.constraints}")
            if spec.source_columns:
                prompt.append(f"- Source columns: {', '.join(spec.source_columns)}")

        prompt.extend([
            "\nDataset Information:",
            f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns",
            f"- Columns: {', '.join(df.columns)}",
            f"- Data Types: {df.dtypes.to_dict()}\n"
        ])

        return "\n".join(prompt)
