import asyncio
from datetime import datetime
from openai import AsyncOpenAI
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import nest_asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, ValidationError, model_validator
import re
from .utils import extract_json, generate_default_prompt
from .exceptions import APIError, DataProcessingError
from IPython.display import display, Markdown
from sentence_transformers import SentenceTransformer

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
        self.conversation_history = []
         # Embedding-based context tracking
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Full conversation history
        self.full_conversation_history: List[Dict[str, Any]] = []
        
        # Contextual summaries with embeddings
        self.context_summaries: List[Dict[str, Any]] = []
        
        # Similarity and context management parameters
        self.context_similarity_threshold = 0.7
        self.max_context_summaries = 5
        self.context_window_tokens = 1000

        if debug:
            logger.setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)

    async def _get_response(self, prompt: str, row_data: Dict[str, Any], feature_names: List[str], system_content: Optional[str] = None) -> str:
        async with self.semaphore:
            try:
                if self.debug:
                    logger.debug(f"Prompt: {prompt}")
                
                # Use a default system content if not provided
                if system_content is None:
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

                if self.debug:
                    print(f"System content: {system_content}")
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
    
    def _generate_context_summary(self, query: str, response: str) -> Dict[str, Any]:
        """
        Generate a contextual summary with embedding
        """
        # Combine query and response for comprehensive context
        context_text = f"{query} | {response}"
        
        # Generate embedding
        context_embedding = self.embedding_model.encode(context_text)
        
        summary = {
            "timestamp": datetime.now(),
            "text": context_text,
            "embedding": context_embedding.tolist(),
            "tokens": len(context_text.split())  # Simple token estimation
        }
        
        return summary
    
    def _calculate_context_relevance(self, new_query: str) -> List[float]:
        """
        Calculate relevance scores of new query to existing context summaries
        
        Returns:
        - List of similarity scores for each existing context summary
        """
        # Encode new query
        new_query_embedding = self.embedding_model.encode(new_query)
        
        # Calculate cosine similarities
        similarities = []
        for summary in self.context_summaries:
            prev_embedding = np.array(summary['embedding'])
            
            # Cosine similarity calculation
            similarity = np.dot(new_query_embedding, prev_embedding) / (
                np.linalg.norm(new_query_embedding) * np.linalg.norm(prev_embedding)
            )
            similarities.append(similarity)
        
        return similarities

    def _prune_context_summaries(self):
        """
        Manage context summaries:
        - Limit number of summaries
        - Remove old or less relevant summaries
        """
        # Sort summaries by timestamp (oldest first)
        self.context_summaries.sort(key=lambda x: x['timestamp'])
        
        # Remove old summaries if we exceed max limit
        while len(self.context_summaries) > self.max_context_summaries:
            self.context_summaries.pop(0)
        
        # Optional: Remove summaries that are very old or low relevance
        current_time = datetime.now()
        self.context_summaries = [
            summary for summary in self.context_summaries
            if (current_time - summary['timestamp']).days < 7  # Keep summaries from last 7 days
        ]

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
        
    def chat(self, query: str, df: pd.DataFrame) -> str:
        """
        Chat method to answer questions about a DataFrame using AI.

        Args:
            query (str): Natural language question about the DataFrame
            df (pd.DataFrame): The DataFrame to analyze

        Returns:
            str: AI-generated response to the query
        """

        similarities = self._calculate_context_relevance(query)
        
        context_text = ""
        if similarities and max(similarities) > self.context_similarity_threshold:
            relevant_indices = [
                i for i, sim in enumerate(similarities) 
                if sim > self.context_similarity_threshold
            ]
            
            # Combine relevant context summaries
            context_text = " | ".join([
                self.context_summaries[i]['text'] 
                for i in relevant_indices
            ])

        df_info = {
            "columns": list(df.columns),
            "shape": df.shape,
            "column_types": df.dtypes.to_dict(),
            "summary_stats": df.describe().to_dict()
        }

        system_content = (
            "You are a precise data analysis assistant. Follow these CRITICAL instructions EXACTLY:\n"
            "1. ALWAYS respond in a VALID JSON format with ONLY an 'answer' key.\n"
            "2. The 'answer' value MUST be a STRING containing your full, complete response.\n"
            "3. DO NOT use any escape characters, newlines, or special formatting in the JSON.\n"
            "4. If you cannot answer the question, return a clear explanation as the answer value.\n"
            "5. Be concise but comprehensive in your analysis.\n"
            "6. Ensure the JSON can be parsed without any errors.\n\n"
            "Conversation Context:\n"
            f"{context_text}\n\n"
            "EXAMPLE VALID RESPONSE:\n"
            '{"answer": "The mean of the Age column is 35.6 years based on the provided DataFrame."}'
        )

        user_content = (
            f"DataFrame Context:\n"
            f"Columns: {df_info['columns']}\n"
            f"Shape: {df_info['shape']}\n"
            f"Column Types: {df_info['column_types']}\n"
            f"Summary Statistics: {df_info['summary_stats']}\n\n"
            f"Question: {query}"
        )

        try:
            response = asyncio.run(self._get_response(
                prompt=user_content, 
                row_data=df_info, 
                feature_names=['answer']
                ,system_content=system_content
            ))

            parsed_response = extract_json(response)
            answer = parsed_response.get('answer', 'I could not generate a response.')

             # Store full conversation history
            full_entry = {
                "timestamp": datetime.now(),
                "query": query,
                "response": answer,
                "df_context": df_info
            }
            self.full_conversation_history.append(full_entry)

            # Generate and store context summary
            context_summary = self._generate_context_summary(query, answer)
            self.context_summaries.append(context_summary)
            
            # Prune context summaries
            self._prune_context_summaries()

            return answer

        except Exception as e:
            error_response = f"An error occurred while processing your query: {str(e)}"
            
            # Store the conversation with the error response
            conversation_entry = {
                "query": query,
                "response": error_response,
                "df_context": df_info
            }
            self.conversation_history.append(conversation_entry)

            return error_response
            

    def get_conversation_history(self, mode='full'):
        """
        Retrieve conversation history
        
        Args:
            mode (str): 'full' or 'summary'
        """
        if mode == 'full':
            return self.full_conversation_history
        elif mode == 'summary':
            return self.context_summaries
        else:
            raise ValueError("Mode must be 'full' or 'summary'")

    def clear_conversation_history(self, mode='all'):
        """
        Clear conversation history
        
        Args:
            mode (str): 'full', 'summary', or 'all'
        """
        if mode in ['full', 'all']:
            self.full_conversation_history.clear()
        if mode in ['summary', 'all']:
            self.context_summaries.clear()