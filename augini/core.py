import asyncio
from openai import AsyncOpenAI
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import nest_asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, ValidationError, root_validator
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

    @root_validator(pre=True)
    def check_prompt(cls, values):
        prompt = values.get('prompt')
        available_columns = values.get('available_columns')
        placeholders = re.findall(r'{(.*?)}', prompt)
        for ph in placeholders:
            if ph not in available_columns:
                raise ValueError(f"Feature '{ph}' used in custom prompt does not exist in available columns: {available_columns}")
        return values


class Augini:
    def __init__(
        self,
        api_key: str,
        use_openrouter: bool = True,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.8,
        max_tokens: int = 500,
        concurrency_limit: int = 10,
        base_url: str = "https://openrouter.ai/api/v1",
        debug: bool = False
    ):
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
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.debug = debug

        if debug:
            logger.setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)

    async def _get_response(self, prompt: str, row_data: Dict[str, Any], feature_names: List[str]) -> str:
        async with self.semaphore:
            try:
                if self.debug:
                    logger.debug(f"Prompt: {prompt}")
                
                json_template = "{" + ", ".join(f'"{name}": "FILL"' for name in feature_names) + "}"
                system_content = (
                    "You are a helpful and very creative assistant that generates hyperrealistic (but fictional) synthetic tabular data based on limited information. "
                    "Ensure the response is a valid JSON object as it is very important."
                )
                user_content = (
                    f"{prompt}\n\n"
                    f"Here is the row data: {row_data}\n\n"
                    f"Please fill the following JSON template with appropriate values:\n{json_template}"
                )

                if self.debug:
                    print(f"System content: {user_content}")
                    logger.debug(f"User content: {user_content}")


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
                
                response_content = response.choices[0].message.content.strip()
                if self.debug:
                    logger.debug(f"Response: {response_content}")
                return response_content
            except Exception as e:
                logger.error(f"Error: {e}")
                raise APIError(f"API request failed: {str(e)}")

    async def _generate_features(self, df: pd.DataFrame, feature_names: List[str], prompt_template: str) -> pd.DataFrame:
        async def process_row(index: int, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
            try:
                row_data = row.to_dict()
                prompt = prompt_template.format(**row_data)
                logger.debug(f"Processing row {index}: {row_data}")
                response = await self._get_response(prompt, row_data, feature_names)
                logger.debug(f"Response for row {index}: {response}")
                feature_values = extract_json(response)
                logger.debug(f"Extracted features for row {index}: {feature_values}")
                if not feature_values or not all(feature in feature_values for feature in feature_names):
                    raise DataProcessingError(f"Expected features are missing in the JSON response: {feature_values}")
                return index, feature_values
            except Exception as e:
                logger.warning(f"Error processing row {index}: {e}")
                return index, {feature: np.nan for feature in feature_names}

        tasks = [process_row(index, row) for index, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
        
        # Sort results by index
        sorted_results = sorted(results, key=lambda x: x[0])
        
        for feature in feature_names:
            df[feature] = [result[1].get(feature, np.nan) for result in sorted_results]
        return df

    def _generate_features_sync(self, df: pd.DataFrame, feature_names: List[str], prompt_template: str) -> pd.DataFrame:
        results = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating features"):
            row_data = row.to_dict()
            prompt = prompt_template.format(**row_data)
            response = asyncio.run(self._get_response(prompt, row_data, feature_names))
            feature_values = extract_json(response)
            results.append(feature_values)
        
        for feature in feature_names:
            df[feature] = [result.get(feature, np.nan) for result in results]
        return df

    def augment_columns(self, df: pd.DataFrame, columns: List[str], custom_prompt: Optional[str] = None, use_sync: bool = False) -> pd.DataFrame:
        result_df = df.copy()
        available_columns = list(result_df.columns)
        column_names = columns

        if custom_prompt:
            try:
                CustomPromptModel(column_names=column_names, prompt=custom_prompt, available_columns=available_columns)
            except ValidationError as e:
                raise ValueError(f"Custom prompt validation error: {e}")

        prompt_template = custom_prompt or generate_default_prompt(column_names, available_columns)
        
        if use_sync:
            return self._generate_features_sync(result_df, column_names, prompt_template)
        else:
            return asyncio.run(self._generate_features(result_df, column_names, prompt_template))

    def augment_single(self, df: pd.DataFrame, column_name: str, custom_prompt: Optional[str] = None, use_sync: bool = False) -> pd.DataFrame:
        result_df = df.copy()
        available_columns = list(result_df.columns)

        if custom_prompt:
            try:
                CustomPromptModel(column_names=[column_name], prompt=custom_prompt, available_columns=available_columns)
            except ValidationError as e:
                raise ValueError(f"Custom prompt validation error: {e}")

        prompt_template = custom_prompt or generate_default_prompt([column_name], available_columns)
        
        if use_sync:
            return self._generate_features_sync(result_df, [column_name], prompt_template)
        else:
            return asyncio.run(self._generate_features(result_df, [column_name], prompt_template))