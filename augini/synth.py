import asyncio
from typing import List, Optional, Dict, Any, Type
from dataclasses import dataclass
from openai import AsyncOpenAI
import json

class Synth:
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
        Initialize the Synth class for general-purpose synthetic data generation.

        Args:
            model (str): Model name to use for generation.
            api_key (str): API key for OpenAI/OpenRouter.
            temperature (float): Temperature for generation.
            max_tokens (int): Maximum tokens for generation.
            use_openrouter (bool): Whether to use OpenRouter API.
            base_url (str): Base URL for OpenRouter API.
            max_concurrent (int): Maximum concurrent API calls.
            debug (bool): Enable debug logging.
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

    async def _get_response(self, prompt: str, schema: Type) -> Any:
        """Get response from the API with proper error handling"""
        async with self.semaphore:
            try:
                system_content = (
                    f"You are a helpful assistant that generates synthetic data based on the following schema: {schema.__doc__}. "
                    "Ensure the response is a valid JSON object as it is very important."
                )

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )

                # Parse the response into the schema class
                response_json = json.loads(response.choices[0].message.content.strip())
                return schema(**response_json)

            except Exception as e:
                raise RuntimeError(f"API request failed: {str(e)}")

    async def synthesize(self, prompt: str, schema: Type) -> Any:
        """
        Generate synthetic data based on the prompt and schema.

        Args:
            prompt (str): The prompt for generating data.
            schema (Type): The schema class for the task.

        Returns:
            Any: An instance of the schema class containing the generated data.
        """
        return await self._get_response(prompt, schema)

    async def get_critique(self, data: Any, critique_schema: Type, critique_prompt: str) -> Any:
        """
        Generate a critique for the generated data.

        Args:
            data (Any): The generated data to critique.
            critique_schema (Type): The schema class for the critique.
            critique_prompt (str): The prompt for generating the critique.

        Returns:
            Any: An instance of the critique schema class containing the critique.
        """
        return await self._get_response(critique_prompt.format(data=data), critique_schema)