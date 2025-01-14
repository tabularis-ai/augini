import asyncio
from typing import List, Optional, Dict, Any, Type
from openai import AsyncOpenAI
import json
import inspect

class SynthError(Exception):
    """Base exception class for Synth errors"""
    pass

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
        debug: bool = False,
        prompt_template: str = None
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
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.debug = debug
        self.prompt_template = prompt_template

    def _validate_schema_class(self, schema: Type) -> None:
        """Validate that the schema class has required attributes and docstring"""
        if not inspect.isclass(schema):
            raise SynthError("Schema must be a class")
            
        if not schema.__doc__:
            raise SynthError("Schema class must have a docstring describing the task")
            
        init_params = inspect.signature(schema.__init__).parameters
        if len(init_params) < 2:  # Excluding self
            raise SynthError("Schema class must have at least one attribute")

    def _validate_examples(self, examples: List[Any], schema: Type) -> None:
        """Validate that examples match the schema"""
        if not examples:
            return
            
        if not isinstance(examples, list):
            raise SynthError("Examples must be provided as a list")
            
        for example in examples:
            if not isinstance(example, schema):
                raise SynthError(
                    f"Example {example} is not an instance of {schema.__name__}"
                )

    def _extract_schema_fields(self, schema: Type) -> Dict[str, Type]:
        """Extract field names and types from schema class"""
        init_params = inspect.signature(schema.__init__).parameters
        return {
            name: param.annotation 
            for name, param in init_params.items() 
            if name != 'self'
        }

    def _format_examples(self, examples: List[Any]) -> str:
        """Format examples for inclusion in the prompt"""
        if not examples:
            return ""
            
        examples_str = "\nExamples:\n"
        for i, example in enumerate(examples, 1):
            examples_str += f"Example {i}:\n{str(example)}\n"
        return examples_str

    async def _get_response(
        self, 
        prompt: str, 
        schema: Type,
        examples: List[Any] = None,
        topics: Dict[str, str] = None
    ) -> Any:
        """Get response from the API and convert to schema instance"""
        async with self.semaphore:
            try:
                # Validate inputs
                self._validate_schema_class(schema)
                self._validate_examples(examples, schema)
                
                # Format prompt if topics provided
                final_prompt = prompt.format(**topics) if topics else prompt
                
                # Extract schema information
                fields = self._extract_schema_fields(schema)
                field_info = "\n".join(f"- {name}: {annotation}" 
                                     for name, annotation in fields.items())
                
                # Create system message with schema info and examples
                system_content = (
                    f"You are a helpful assistant that generates data following this schema:\n"
                    f"Task: {schema.__doc__}\n"
                    f"Fields:\n{field_info}"
                )
                
                if examples:
                    system_content += self._format_examples(examples)
                    system_content += "\nGenerate new data following the same patterns shown in the examples."
                
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
                
                # Parse response and create schema instance
                try:
                    response_json = json.loads(response.choices[0].message.content.strip())
                    return schema(**response_json)
                except (json.JSONDecodeError, TypeError) as e:
                    raise SynthError(f"Failed to parse API response: {str(e)}")
                except Exception as e:
                    raise SynthError(f"Failed to create schema instance: {str(e)}")
            
            except Exception as e:
                if isinstance(e, SynthError):
                    raise
                raise SynthError(f"API request failed: {str(e)}")

    async def synthesize(
        self, 
        prompt: str = None, 
        schema: Type = None,
        examples: List[Any] = None,
        topics: Dict[str, str] = None
    ) -> Any:
        """
        Generate synthetic data based on the prompt, schema, and optional examples.
        
        Args:
            prompt: The prompt or template for generating data
            schema: The schema class defining the data structure
            examples: Optional list of example instances of the schema class
            topics: Dictionary of variables to format the prompt template
            
        Returns:
            An instance of the schema class
        """
        if schema is None:
            raise ValueError("Schema must be provided")
            
        if prompt is None and self.prompt_template is None:
            raise ValueError("Either prompt or instance prompt_template must be provided")
            
        use_prompt = prompt if prompt is not None else self.prompt_template
        return await self._get_response(use_prompt, schema, examples, topics)

    async def get_critique(
        self, 
        data: Any, 
        critique_schema: Type, 
        critique_prompt: str,
        examples: List[Any] = None,
        topics: Dict[str, str] = None
    ) -> Any:
        """Generate a critique for the generated data with optional examples"""
        format_dict = {"data": str(data)}
        if topics:
            format_dict.update(topics)
            
        return await self._get_response(
            critique_prompt,
            critique_schema,
            examples,
            format_dict
        )