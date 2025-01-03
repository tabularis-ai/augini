import asyncio
from openai import AsyncOpenAI
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import nest_asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, model_validator
import re
from .utils import extract_json, generate_default_prompt
from .exceptions import APIError, DataProcessingError

nest_asyncio.apply()

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

class CustomPromptModel(BaseModel):
    column_names: List[str]
    prompt: str
    available_columns: List[str]

    @model_validator(mode="before")
    def check_prompt(cls, values):
        prompt = values.get('prompt')
        available_columns = values.get('available_columns')
        placeholders = re.findall(r'{(.*?)}', prompt)
        for ph in placeholders:
            if ph not in available_columns:
                raise ValueError(f"Feature '{ph}' used in custom prompt does not exist in available columns: {available_columns}")
        return values


class Augment:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        use_openrouter: bool = True,
        base_url: str = "https://openrouter.ai/api/v1",
        max_concurrent: int = 5,
        debug: bool = False
    ):
        """
        Initialize the Augment class.
        
        Args:
            model (str): Model name to use for generation
            api_key (str): API key for OpenAI/OpenRouter
            temperature (float): Temperature for generation
            max_tokens (int): Maximum tokens for generation
            use_openrouter (bool): Whether to use OpenRouter API
            base_url (str): Base URL for OpenRouter API
            max_concurrent (int): Maximum concurrent API calls
            debug (bool): Enable debug logging
        """
        if use_openrouter:
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        else:
            self.client = AsyncOpenAI(api_key=api_key)

        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.debug = debug

        if debug:
            logger.setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)

    async def _get_response(self, prompt: str, row_data: Dict[str, Any], feature_names: List[str]) -> str:
        """Get response from OpenAI API with proper error handling"""
        async with self.semaphore:
            try:
                system_content = (
                    "You are a helpful and very creative assistant that generates hyperrealistic (but fictional) synthetic tabular data based on limited information. "
                    "Ensure the response is a valid JSON object as it is very important."
                )
                
                json_template = "{" + ", ".join(f'"{name}": "FILL"' for name in feature_names) + "}"
                
                user_content = (
                    f"{prompt}\n\n"
                    f"Here is the row data: {row_data}\n\n"
                    f"Please fill the following JSON template with appropriate values:\n{json_template}"
                )

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens, 
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error: {e}")
                if "401" in str(e) or "authentication" in str(e).lower():
                    raise APIError("Authentication failed: Invalid API key") from e
                raise APIError(f"API request failed: {str(e)}") from e

    async def _generate_features(self, df: pd.DataFrame, feature_names: List[str], prompt_template: str) -> pd.DataFrame:
        """Generate features for all rows with proper resource management"""
        async def process_row(index: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
            try:
                row_data = row.to_dict()
                prompt = prompt_template.format(**row_data)
                response = await self._get_response(prompt, row_data, feature_names)
                feature_values = extract_json(response)
                if not feature_values or not all(feature in feature_values for feature in feature_names):
                    raise DataProcessingError(f"Expected features are missing in the JSON response: {feature_values}")
                return index, feature_values
            except APIError:
                raise
            except Exception as e:
                logger.warning(f"Error processing row {index}: {e}")
                return index, {feature: np.nan for feature in feature_names}

        tasks = [process_row(index, row) for index, row in df.iterrows()]
        try:
            results = await asyncio.gather(*tasks)
            sorted_results = sorted(results, key=lambda x: x[0])
            
            for feature in feature_names:
                df[feature] = [result[1].get(feature, np.nan) for result in sorted_results]
            return df
        except APIError:
            raise

    def augment_single(self, df: pd.DataFrame, column_name: str, custom_prompt: Optional[str] = None) -> pd.DataFrame:
        """
        Augment a single column in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column_name (str): Name of the column to generate
            custom_prompt (Optional[str]): Custom prompt template
            
        Returns:
            pd.DataFrame: DataFrame with the new augmented column
        """
        result_df = df.copy()
        available_columns = list(result_df.columns)
        prompt = custom_prompt or generate_default_prompt([column_name], available_columns)
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._generate_features(result_df, [column_name], prompt))
            return result
        finally:
            loop.close()

    def augment_columns(self, df: pd.DataFrame, columns: List[str], custom_prompt: Optional[str] = None) -> pd.DataFrame:
        """
        Augment multiple columns in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): List of column names to generate
            custom_prompt (Optional[str]): Custom prompt template
            
        Returns:
            pd.DataFrame: DataFrame with the new augmented columns
        """
        result_df = df.copy()
        available_columns = list(result_df.columns)
        prompt = custom_prompt or generate_default_prompt(columns, available_columns)
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._generate_features(result_df, columns, prompt))
            return result
        finally:
            loop.close() 