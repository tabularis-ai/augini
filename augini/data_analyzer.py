from typing import Optional, Dict, Any
import pandas as pd
from pydantic import BaseModel, Field
import openai
from .config import AuginiConfig
import asyncio
from datetime import datetime
import numpy as np
from tqdm.auto import tqdm
import nest_asyncio
import logging
from typing import List, Tuple
from openai import AsyncOpenAI
from .utils import extract_json, configure_logging
from .exceptions import APIError, DataProcessingError

nest_asyncio.apply()

logger = logging.getLogger(__name__)

class ChatResponse(BaseModel):
    """Model for chat response data."""

    analysis: str = Field(..., description="Main analysis text")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DataAnalyzer:
    """AI-powered data analysis assistant."""

    def __init__(
        self,
        api_key: str = None,
        config: Optional[AuginiConfig] = None,
        **kwargs
    ):
        """Initialize the analyzer.

        Args:
            api_key: API key for authentication
            config: AuginiConfig instance (optional)
            **kwargs: Override config values (e.g., base_url, model, etc.)
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
            base_url=self.config.base_url  # If None, uses OpenAI's default URL
        )

        # Configure logging
        if not self.config.debug:
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("augini").setLevel(logging.WARNING)

        # Initialize data caching attributes
        self._data_hash = None
        self._data_context_cache = None

        # Initialize semaphore for concurrent API calls
        self._semaphore = asyncio.Semaphore(self.config.concurrency_limit)

    def fit(self, df: pd.DataFrame) -> "DataAnalyzer":
        """Prepare data for analysis.

        Args:
            df: DataFrame to analyze

        Returns:
            self for method chaining
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame must not be empty")

        self._df = df
        return self

    def _prepare_context(self, query: str) -> str:
        """Prepare data context for the model.

        Args:
            query: User's analysis question

        Returns:
            Formatted context string
        """
        df = self._df

        context = [
            f"Query: {query}\n",
            "Dataset Information:",
            f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns",
            f"- Columns: {', '.join(df.columns)}",
            f"- Data Types: {df.dtypes.to_dict()}\n",
            "Sample Data (first 5 rows):",
            f"{df.head().to_string()}\n",
            "Column Statistics:",
        ]

        for col in df.columns:
            context.append(f"\n{col}:")
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numerical column stats
                desc = df[col].describe()
                context.append(f"- Type: numeric")
                context.append(f"- Mean: {desc['mean']:.2f}")
                context.append(f"- Median: {desc['50%']:.2f}")
                context.append(f"- Std: {desc['std']:.2f}")
                context.append(f"- Range: [{desc['min']:.2f}, {desc['max']:.2f}]")
                context.append(f"- Unique Values: {df[col].nunique()}")
            else:
                # Categorical column stats
                value_counts = df[col].value_counts()
                context.append(f"- Type: categorical")
                context.append(f"- Unique Values: {df[col].nunique()}")
                context.append(f"- Top Values: {dict(value_counts.head(3))}")

        return "\n".join(context)

    async def _get_response(
        self,
        prompt: str,
        row_data: Dict[str, Any],
        feature_names: List[str],
        system_content: Optional[str] = None,
    ) -> str:
        async with self._semaphore:
            try:
                if self.config.debug:
                    logger.debug(f"Prompt: {prompt}")

                # Use a default system content if not provided
                if system_content is None:
                    system_content = (
                        "You are an expert data analyst assistant specialized in tabular data analysis. "
                        "Your response must be a valid, complete JSON object with a single 'answer' key containing markdown-formatted text. "
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

    def _generate_smart_data_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data context from DataFrame."""
        # Basic DataFrame info
        basic_info = {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Generate statistics for all columns
        column_stats = {}

        for col in df.columns:
            col_stats = {}

            if df[col].dtype in ["int64", "float64"]:
                # Numerical column stats
                desc = df[col].describe()
                col_stats.update(
                    {
                        "type": "numeric",
                        "mean": desc["mean"],
                        "median": df[col].median(),
                        "std": desc["std"],
                        "range": [desc["min"], desc["max"]],
                        "missing": df[col].isna().sum(),
                        "distribution": {
                            "quartiles": [desc["25%"], desc["50%"], desc["75%"]],
                            "skew": df[col].skew(),
                        },
                    }
                )
            else:
                # Categorical column stats
                value_counts = df[col].value_counts()
                col_stats.update(
                    {
                        "type": "categorical",
                        "unique_values": len(value_counts),
                        "top_categories": value_counts.head(5).to_dict(),
                        "missing": df[col].isna().sum(),
                    }
                )

            column_stats[col] = col_stats

        # Calculate correlations between numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        correlations = {}

        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    correlation = corr_matrix.loc[col1, col2]
                    if abs(correlation) > 0.3:  # Only include meaningful correlations
                        correlations[f"{col1}_vs_{col2}"] = correlation

        # Add data samples
        samples = {
            "head": df.head(3).to_dict(orient="records"),
            "random": df.sample(n=min(3, len(df))).to_dict(orient="records"),
        }

        # Add data quality metrics
        data_quality = {
            "total_missing": df.isna().sum().sum(),
            "missing_by_column": df.isna().sum().to_dict(),
            "duplicated_rows": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # in MB
        }

        return {
            "basic_info": basic_info,
            "column_stats": column_stats,
            "correlations": correlations,
            "samples": samples,
            "data_quality": data_quality,
        }

    def _calculate_df_hash(self, df: pd.DataFrame) -> str:
        """Calculate a hash for the DataFrame to detect changes"""
        df_info = f"{df.shape}_{list(df.columns)}_{df.head(1).to_json()}_{df.tail(1).to_json()}"
        return str(hash(df_info))

    def _get_smart_data_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get smart data context with caching"""
        current_hash = self._calculate_df_hash(df)

        # Return cached context if DataFrame hasn't changed
        if self._data_hash == current_hash and self._data_context_cache is not None:
            if self.config.debug:
                logger.info("Using cached data context")
            return self._data_context_cache

        # Generate new context if DataFrame has changed
        if self.config.debug:
            logger.info("Generating new data context")

        context = self._generate_smart_data_context(df)

        # Update cache
        self._data_context_cache = context
        self._data_hash = current_hash

        return context

    def chat(self, query: str) -> str:
        """Analyze data using natural language.

        Args:
            query: Analysis question or instruction

        Returns:
            Analysis results as markdown-formatted text

        Raises:
            RuntimeError: If fit() hasn't been called
        """
        if self._df is None:
            raise RuntimeError("Must call fit() before analysis")

        # Use cached data context if available
        df_info = self._get_smart_data_context(self._df)

        user_content = (
            f"DataFrame Analysis Context:\n"
            f"Basic Info:\n"
            f"- Columns: {df_info['basic_info']['columns']}\n"
            f"- Shape: {df_info['basic_info']['shape']}\n"
            f"- Data Types: {df_info['basic_info']['dtypes']}\n\n"
            f"Column Statistics:\n"
            f"{df_info['column_stats']}\n\n"
            f"Correlations:\n"
            f"{df_info['correlations']}\n\n"
            f"Data Quality:\n"
            f"- Total Missing Values: {df_info['data_quality']['total_missing']}\n"
            f"- Duplicated Rows: {df_info['data_quality']['duplicated_rows']}\n"
            f"- Missing by Column: {df_info['data_quality']['missing_by_column']}\n\n"
            f"Sample Data:\n"
            f"First 3 rows: {df_info['samples']['head']}\n\n"
            f"Question: {query}"
        )

        system_content = (
            "You are an expert data analyst assistant specialized in tabular data analysis. "
            "Your response must be a valid, complete JSON object with a single 'answer' key containing markdown-formatted text. "
            "Make sure to properly close all JSON brackets and escape special characters.\n\n"
            "If the question is not clear, ask for clarification. IMPORTANT: If the question is not related to the data, ask for a different question.\n\n"
            "ANALYSIS SCOPE:\n"
            "   - Answer questions about data characteristics, patterns, and quality\n"
            "   - Analyze data distribution, relationships, and anomalies\n"
            "   - Assess data authenticity and potential synthetic patterns\n"
            "   - Consider data collection and generation methods\n"
            "   - Evaluate data consistency and realism\n\n"
            "RESPONSE APPROACH:\n"
            "   - Start with direct answers to the specific question\n"
            "   - Support claims with evidence from the data\n"
            "   - Consider both statistical and qualitative indicators\n"
            "   - Note any limitations or uncertainties\n"
            "   - Use data patterns to inform conclusions\n\n"
            "Example of correctly formatted response:\n"
            '{"answer": "## Analysis Results 🔍\\n\\n**Key Finding:** The data shows interesting patterns\\n\\n### Details\\n- The `column_name` shows X\\n- Statistics indicate Y\\n\\n> Evidence: Z"}'
        )

        try:
            response = asyncio.run(
                self._get_response(
                    prompt=user_content,
                    row_data=df_info,
                    feature_names=["answer"],
                    system_content=system_content,
                )
            )

            try:
                # Clean up the response if it contains markdown-style code blocks
                response = response.replace("```json", "").replace("```", "").strip()
                parsed_response = extract_json(response)

                if parsed_response is None or "answer" not in parsed_response:
                    # If JSON parsing fails or 'answer' key is missing, return the raw response with formatting
                    formatted_response = (
                        "## ⚠️ Response Format Note\n\n"
                        "I received a response but it wasn't in the expected JSON format. "
                        "Here's the raw response:\n\n"
                        "---\n\n"
                        f"{response}\n\n"
                        "---\n\n"
                        "_Please try asking your question again._"
                    )
                    return formatted_response

                answer = parsed_response["answer"]
            except Exception as json_error:
                # Handle JSON parsing errors with a user-friendly message
                formatted_response = (
                    "## ⚠️ Response Processing Error\n\n"
                    "I encountered an error while processing the response. "
                    f"Error details: {str(json_error)}\n\n"
                    "Here's the raw response I received:\n\n"
                    "---\n\n"
                    f"{response}\n\n"
                    "---\n\n"
                    "_Please try rephrasing your question._"
                )
                return formatted_response

            return answer

        except Exception as e:
            error_response = (
                "## ❌ Error Processing Query\n\n"
                f"An error occurred while processing your query: {str(e)}\n\n"
                "_Please try again or rephrase your question._"
            )
            return error_response
