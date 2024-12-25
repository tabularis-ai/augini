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
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
        max_tokens: int = 750,
        concurrency_limit: int = 10,
        base_url: str = "https://openrouter.ai/api/v1",
        debug: bool = False,
        enable_memory: bool = False
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
        
        # Initialize memory-related attributes only if enabled
        self.enable_memory = enable_memory
        self.embedding_model = None
        
        # Initialize all memory-related attributes as None
        self.conversation_history = None
        self.full_conversation_history = None
        self.context_summaries = None
        self.context_similarity_threshold = None
        self.max_context_summaries = None
        self.context_window_tokens = None

        # Add cache for smart data context
        self._data_context_cache = None
        self._data_hash = None

        if enable_memory:
            # Only initialize memory-related attributes if memory is enabled
            self.conversation_history = []
            self.full_conversation_history = []
            self.context_summaries = []
            self.context_similarity_threshold = 0.7
            self.max_context_summaries = 5
            self.context_window_tokens = 1000

        if debug:
            logger.setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)

    def _initialize_memory(self):
        """Lazy initialization of sentence transformer model"""
        if not self.enable_memory:
            raise RuntimeError(
                "Memory features are disabled. Enable them by setting enable_memory=True "
                "and installing required dependencies: pip install augini[memory]"
            )
        
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import spacy  # Also check for spacy
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                raise ImportError(
                    "Memory features require additional dependencies. "
                    "Install them with: pip install augini[memory]"
                )

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
        """Generate a contextual summary with embedding"""
        self._initialize_memory()  # Lazy initialization
        
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
        """Calculate relevance scores of new query to existing context summaries"""
        self._initialize_memory()  # Lazy initialization
        
        # Encode new query
        new_query_embedding = self.embedding_model.encode(new_query)
        
        # Calculate cosine similarities
        similarities = []
        for summary in self.context_summaries:
            prev_embedding = np.array(summary['embedding'])
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
        
    def _generate_smart_data_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data context from DataFrame.
        Includes basic info, statistics, correlations, and data samples.
        """
        # Basic DataFrame info
        basic_info = {
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict()
        }

        # Generate statistics for all columns
        column_stats = {}
        
        for col in df.columns:
            col_stats = {}
            
            if df[col].dtype in ['int64', 'float64']:
                # Numerical column stats
                desc = df[col].describe()
                col_stats.update({
                    "type": "numeric",
                    "mean": desc['mean'],
                    "median": df[col].median(),
                    "std": desc['std'],
                    "range": [desc['min'], desc['max']],
                    "missing": df[col].isna().sum(),
                    "distribution": {
                        'quartiles': [desc['25%'], desc['50%'], desc['75%']],
                        'skew': df[col].skew()
                    }
                })
            else:
                # Categorical column stats
                value_counts = df[col].value_counts()
                col_stats.update({
                    "type": "categorical",
                    "unique_values": len(value_counts),
                    "top_categories": value_counts.head(5).to_dict(),
                    "missing": df[col].isna().sum()
                })

            column_stats[col] = col_stats

        # Calculate correlations between numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        correlations = {}
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    correlation = corr_matrix.loc[col1, col2]
                    if abs(correlation) > 0.3:  # Only include meaningful correlations
                        correlations[f"{col1}_vs_{col2}"] = correlation

        # Add data samples
        samples = {
            "head": df.head(3).to_dict(orient='records'),
            "random": df.sample(n=min(3, len(df))).to_dict(orient='records')
        }

        # Add data quality metrics
        data_quality = {
            "total_missing": df.isna().sum().sum(),
            "missing_by_column": df.isna().sum().to_dict(),
            "duplicated_rows": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # in MB
        }

        return {
            "basic_info": basic_info,
            "column_stats": column_stats,
            "correlations": correlations,
            "samples": samples,
            "data_quality": data_quality
        }

    def _calculate_df_hash(self, df: pd.DataFrame) -> str:
        """Calculate a hash for the DataFrame to detect changes"""
        # Use a combination of shape, columns, and sample of data for quick hash
        df_info = f"{df.shape}_{list(df.columns)}_{df.head(1).to_json()}_{df.tail(1).to_json()}"
        return str(hash(df_info))

    def _get_smart_data_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get smart data context with caching"""
        current_hash = self._calculate_df_hash(df)
        
        # Return cached context if DataFrame hasn't changed
        if self._data_hash == current_hash and self._data_context_cache is not None:
            if self.debug:
                logger.info("Using cached data context")
            return self._data_context_cache
        
        # Generate new context if DataFrame has changed
        if self.debug:
            logger.info("Generating new data context")
            
        context = self._generate_smart_data_context(df)
        
        # Update cache
        self._data_context_cache = context
        self._data_hash = current_hash
        
        return context

    def chat(self, query: str, df: pd.DataFrame, use_memory: bool = False) -> str:
        """
        Chat method with smart data context generation.
        """
        # Initialize memory-related attributes if they're None
        if use_memory:
            if self.full_conversation_history is None:
                self.full_conversation_history = []
            if self.context_summaries is None:
                self.context_summaries = []

        context_text = ""
        if use_memory:
            try:
                self._initialize_memory()
                similarities = self._calculate_context_relevance(query)
                if similarities and max(similarities) > self.context_similarity_threshold:
                    relevant_indices = [
                        i for i, sim in enumerate(similarities) 
                        if sim > self.context_similarity_threshold
                    ]
                    context_text = " | ".join([
                        self.context_summaries[i]['text'] 
                        for i in relevant_indices
                    ])
            except (RuntimeError, ImportError) as e:
                logger.warning(f"Memory features unavailable: {str(e)}")

        # Use cached data context if available
        df_info = self._get_smart_data_context(df)

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
            
            # "1. RESPONSE FORMAT REQUIREMENTS:\n"
            # "   - Start with '{\"answer\": \"' exactly\n"
            # "   - End with '\"}' exactly\n"
            # "   - Escape all quotes within the answer text\n"
            # "   - Keep responses concise and focused\n"
            # "   - Ensure JSON is properly closed\n\n"
            
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
            
            # "4. MARKDOWN FORMATTING:\n"
            # "   - Use markdown for clear structure\n"
            # "   - Bold key findings with **text**\n"
            # "   - Use `backticks` for column names\n"
            # "   - Add relevant emojis for emphasis\n"
            # "   - Include evidence in > blockquotes\n\n"
            
            "Example of correctly formatted response:\n"
            '{\"answer\": \"## Analysis Results 🔍\\n\\n**Key Finding:** The data shows interesting patterns\\n\\n### Details\\n- The `column_name` shows X\\n- Statistics indicate Y\\n\\n> Evidence: Z\"}'
            
            + (f"\n\nPrevious Conversation Context:\n{context_text}" if context_text else "")
        )

        try:
            response = asyncio.run(self._get_response(
                prompt=user_content, 
                row_data=df_info, 
                feature_names=['answer'],
                system_content=system_content
            ))

            try:
                # Clean up the response if it contains markdown-style code blocks
                response = response.replace("```json", "").replace("```", "").strip()
                parsed_response = extract_json(response)
                
                if parsed_response is None or 'answer' not in parsed_response:
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
                
                answer = parsed_response['answer']
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

            # Store full conversation history
            if use_memory and self.full_conversation_history is not None:
                full_entry = {
                    "timestamp": datetime.now(),
                    "query": query,
                    "response": answer,
                    "df_context": df_info
                }
                self.full_conversation_history.append(full_entry)

                # Generate and store context summary
                context_summary = self._generate_context_summary(query, answer)
                if self.context_summaries is not None:
                    self.context_summaries.append(context_summary)
                    # Prune context summaries
                    self._prune_context_summaries()

            return answer

        except Exception as e:
            error_response = (
                "## ❌ Error Processing Query\n\n"
                f"An error occurred while processing your query: {str(e)}\n\n"
                "_Please try again or rephrase your question._"
            )
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