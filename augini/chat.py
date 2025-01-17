import json
import logging
from datetime import datetime
from openai import OpenAI
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import markdown

from augini.tools import AVAILABLE_TOOLS, DataAnalysisTools
from .exceptions import APIError
from .utils import extract_json
import ipywidgets as widgets
from IPython.display import display, HTML, Markdown

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

class Chat:
    def __init__(
        self,
        df: pd.DataFrame,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        temperature: float = 0.2,
        use_openrouter: bool = True,
        base_url: str = "https://openrouter.ai/api/v1",
        debug: bool = False,
        enable_memory: bool = False,
        interactive: bool = False  
    ):
        if not api_key:
            raise APIError("API key is required. Please provide a valid API key.")
        
        if use_openrouter:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        else:
            self.client = OpenAI(api_key=api_key)

        self.model_name = model
        self.temperature = temperature
        self.df = df
        self.debug = debug
        
        # Initialize memory-related attributes
        self.enable_memory = enable_memory
        self.embedding_model = None
        self.conversation_history = None
        self.full_conversation_history = None
        self.context_summaries = None
        self.context_similarity_threshold = None
        self.max_context_summaries = None
        self.context_window_tokens = None

        # Add cache for smart data context
        self._data_context_cache = None
        self._data_hash = None

        self.interactive = interactive
        self._interactive_components = {
            'text_input': None,
            'send_button': None,
            'clear_button': None,
            'export_button': None,
            'exit_button': None,
            'chat_output': None,
            'chat_history': [],
            'container': None
        }

        if enable_memory:
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


    def _setup_interactive_interface(self):
        """Set up the interactive chat interface components"""

         # Add a loading widget
        self._interactive_components['loading'] = widgets.HTML(
            value="<div style='text-align: center; margin: 20px;'>"
                "<i class='fa fa-spinner fa-spin' style='font-size: 24px; color: #4a90e2;'></i>"
                "<div style='margin-top: 10px; color: #666;'>Loading...</div>"
                "</div>"
        )

        self._interactive_components['loading'].layout.visibility = 'hidden'  # Hide by default
        
        self._interactive_components = {
            'text_input': widgets.Text(
                placeholder='Type your question here...',
                description='Query:',
                layout=widgets.Layout(width='90%')
            ),
            'send_button': widgets.Button(
                description='Send',
                button_style='primary',
                icon='paper-plane',
                layout=widgets.Layout(width='100px')
            ),
            'clear_button': widgets.Button(
                description='Clear Chat',
                button_style='warning',
                icon='trash',
                layout=widgets.Layout(width='100px')
            ),
            'export_button': widgets.Button(
                description='Export Chat',
                button_style='info',
                icon='download',
                layout=widgets.Layout(width='100px')
            ),
            'exit_button': widgets.Button(
                description='Exit Chat',
                button_style='danger',
                icon='sign-out',
                layout=widgets.Layout(width='100px')
            ),
            'chat_output': widgets.Output(
                layout=widgets.Layout(
                    border='1px solid #ddd',
                    padding='10px',
                    height='500px',
                    overflow_y='auto',
                    width='100%',
                    margin='10px 0'
                )
            ),
            'loading': widgets.HTML(
                value="<div style='text-align: center; margin: 20px;'>"
                    "<i class='fa fa-spinner fa-spin' style='font-size: 24px; color: #4a90e2;'></i>"
                    "<div style='margin-top: 10px; color: #666;'>Loading...</div>"
                    "</div>",
                layout=widgets.Layout(visibility='hidden')  # Hide by default
            ),
            'chat_history': []
        }

        # Create input container at top
        input_container = widgets.HBox(
            [self._interactive_components['text_input'], 
            self._interactive_components['send_button']],
            layout=widgets.Layout(
                width='100%',
                align_items='center',
                margin='10px 0'
            )
        )

        # Create button panel below input
        button_panel = widgets.HBox(
            [self._interactive_components['clear_button'],
            self._interactive_components['export_button'],
            self._interactive_components['exit_button']],
            layout=widgets.Layout(
                width='100%',
                justify_content='flex-end',
                margin='10px 0'
            )
        )

        # Create main container with top-down layout
        main_container = widgets.VBox(
            [input_container,
            button_panel,
            self._interactive_components['loading'],
            self._interactive_components['chat_output']],
            layout=widgets.Layout(
                border='1px solid #ccc',
                padding='10px',
                width='80%',
                margin='0 auto'
            )
        )

        self._interactive_components['container'] = main_container

        # Attach event handlers
        self._interactive_components['send_button'].on_click(self._handle_send)
        self._interactive_components['clear_button'].on_click(self._handle_clear)
        self._interactive_components['export_button'].on_click(self._handle_export)
        self._interactive_components['exit_button'].on_click(self._handle_exit)
        self._interactive_components['text_input'].on_submit(self._handle_send)

        # Display the container
        display(main_container)

        # Display welcome message
        with self._interactive_components['chat_output']:
            display(HTML(
                "<div style='color: #666; padding: 10px; text-align: center;'>"
                "👋 Welcome to Interactive Chat! Ask me anything about your data."
                "<br>Type your question and press Enter or click Send."
                "<br>Click 'Clear Chat' to reset the conversation or 'Exit Chat' to end."
                "</div>"
            ))
            
    def _format_message(self, content: str, is_user: bool = False) -> str:
        """Format chat messages with improved styling and markdown support.
        
        Args:
            content (str): The message content (HTML from the model or plain text from the user).
            is_user (bool): Whether the message is from the user.
        
        Returns:
            str: Formatted message as HTML for display.
        """
        # Convert markdown to HTML for non-user messages (model responses)
        if not is_user:
            html_content = markdown.markdown(content)
            content = html_content

        # Define message container style
        container_style = f"""
            padding: 12px;
            margin: 8px 0;
            border-radius: 12px;
            max-width: 70%;
            word-wrap: break-word;
            {"margin-left: auto; background-color: #e3f2fd;" if is_user else "margin-right: auto; background-color: #f5f5f5;"}
        """
        
        # Define message layout
        if is_user:
            return f"""
                <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                    <div style='{container_style}'>
                        <div style='font-weight: bold; margin-bottom: 5px;'>You:</div>
                        <div>{content}</div>
                    </div>
                </div>
            """
        else:
            return f"""
                <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                    <div style='{container_style}'>
                        <div style='font-weight: bold; margin-bottom: 5px;'>Augini:</div>
                        <div>{content}</div>
                    </div>
                </div>
            """

    def _add_to_history(self, role: str, content: str):
        """Add a message to chat history with timestamp"""
        if self.interactive:
            self._interactive_components['chat_history'].append({
                'timestamp': datetime.now().isoformat(),
                'role': role,
                'content': content
            })

    def _handle_send(self, _):
        """Handle send button click or input submission in interactive mode"""
        if not self.interactive:
            return

        query = self._interactive_components['text_input'].value.strip()
        if not query:
            return

        # Clear input
        self._interactive_components['text_input'].value = ''

        # Display user message and add to history
        with self._interactive_components['chat_output']:
            display(HTML(self._format_message(query, is_user=True)))
        self._add_to_history('user', query)

        # Show loading indicator
        self._interactive_components['loading'].layout.visibility = 'visible'

        try:
            # Get response using the existing chat functionality
            response = self._get_chat_response(query)

            # Display assistant message with markdown support
            with self._interactive_components['chat_output']:
                display(HTML(self._format_message(response)))
            self._add_to_history('assistant', response)


        except Exception as e:
            error_msg = f"**Error:** {str(e)}"
            with self._interactive_components['chat_output']:
                display(HTML(self._format_message(error_msg)))
            self._add_to_history('system', f"Error: {str(e)}")

        finally:
            # Hide loading indicator
            self._interactive_components['loading'].layout.visibility = 'hidden'


    def _handle_clear(self, _):
        """Handle clear button click in interactive mode"""
        if not self.interactive:
            return

        self._interactive_components['chat_history'] = []
        self._interactive_components['chat_output'].clear_output()

        # Display welcome message again
        with self._interactive_components['chat_output']:
            display(HTML(
                "<div style='color: #666; padding: 10px;'>"
                "Chat cleared! Ask me a new question about your data."
                "</div>"
            ))

    def _handle_export(self, _):
        """Handle export button click in interactive mode"""
        if not self.interactive:
            return

        if not self._interactive_components['chat_history']:
            with self._interactive_components['chat_output']:
                display(HTML(
                    "<div style='color: orange; padding: 10px;'>"
                    "No chat history to export!"
                    "</div>"
                ))
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.md"

        with open(filename, 'w') as f:
            f.write("# Chat History\n\n")
            for msg in self._interactive_components['chat_history']:
                time = datetime.fromisoformat(msg['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                role = msg['role'].title()
                content = msg['content']
                f.write(f"## {role} - {time}\n\n{content}\n\n---\n\n")

        with self._interactive_components['chat_output']:
            display(HTML(
                f"<div style='color: green; padding: 10px;'>"
                f"Chat history exported to {filename}"
                f"</div>"
            ))

    def _handle_exit(self, _):
        """Handle exit button click in interactive mode"""
        if not self.interactive:
            return

        # Clear the chat output and close the widget
        self._interactive_components['chat_output'].clear_output()
        self._interactive_components['container'].close()

        # Display exit message
        with self._interactive_components['chat_output']:
            display(HTML(
                "<div style='color: #666; padding: 10px;'>"
                "👋 Chat session ended. Thank you for using Interactive Chat!"
                "</div>"
            ))

    def __call__(self, query: str) -> str:
        """
        Ask a question about the DataFrame.
        
        Args:
            query (str): The question about the DataFrame
            
        Returns:
            str: The response from the model
        """
        return self._get_chat_response(query)

    def start_interactive(self):
        """Start the interactive chat session"""
        if not self.interactive:
            raise RuntimeError("Interactive mode is not enabled. Initialize the class with interactive=True")
        
        # Set up and display the interface
        self._setup_interactive_interface()


    def get_chat_history(self, mode='full'):
        """
        Retrieve conversation history
        
        Args:
            mode (str): 'full', 'summary', or 'interactive'
        """
        if mode == 'interactive' and self.interactive:
            return self._interactive_components['chat_history']
        elif mode == 'full':
            return self.full_conversation_history
        elif mode == 'summary':
            return self.context_summaries
        else:
            raise ValueError("Mode must be 'full', 'summary', or 'interactive'")

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
    

    def _format_tool_results(self, tool_results: List[Dict]) -> str:
        """Format tool results for inclusion in the response"""
        formatted_results = []
        
        for result in tool_results:
            # Format code execution results
            if result["tool"] == "execute_code":
                formatted_result = f"```python\n{result['args']['code']}\n```\n\n"
                
                # Add output if present
                if result.get("result"):
                    formatted_result += f"**Output:**\n```\n{result['result']}\n```\n\n"
                
                formatted_results.append(formatted_result)
            else:
                # Format other tool results
                formatted_results.append(
                    f"**{result['tool']} Results:**\n```\n{result['result']}\n```\n"
                )
        
        return "\n".join(formatted_results)
                


    def _get_chat_response(self, query: str) -> str:
        try:

            # Initialize tools
            self.tools = DataAnalysisTools(self.df)
            
            # Get smart data context
            df_info = self._get_smart_data_context(self.df)

            # Get memory context if enabled
            context_text = ""
            if self.enable_memory:
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
                "Your response must be a valid, complete JSON object with a single 'answer' key "
                "containing a markdown fragment. The markdown should be well-formatted and use "
                "appropriate markdown syntax such as headings (#, ##), lists (-, *), code blocks (```), "
                "and inline code (`) for clarity and readability. Do not include HTML tags—only markdown "
                "content is needed. Ensure that all special characters in the JSON string are properly escaped.\n\n"
                "If the question is not clear, ask for clarification. IMPORTANT: If the question is not "
                "related to the data, ask for a different question.\n\n"

                "MARKDOWN STYLING GUIDELINES:\n"
                "- Use `# Heading` for main headings\n"
                "- Use `## Subheading` for subheadings\n"
                "- Use `-` or `*` for lists\n"
                "- Use `**bold**` for emphasis\n"
                "- Use `*italic*` for subtle emphasis\n"
                "- Use ``` for code blocks\n"
                "- Use `inline code` for code snippets\n\n"
                
                "ANALYSIS SCOPE:\n"
                "   - Answer questions about data characteristics, patterns, and quality\n"
                "   - Analyze data distribution, relationships, and anomalies\n"
                "   - Assess data authenticity and potential synthetic patterns\n"
                "   - Consider data collection and generation methods\n"
                "   - Evaluate data consistency and realism\n\n"
                "   - Write safe pandas code to handle analysis and plots otherwise\n\n"
                
                "RESPONSE APPROACH:\n"
                "   - Start with direct answers to the specific question\n"
                "   - Support claims with evidence from the data\n"
                "   - Consider both statistical and qualitative indicators\n"
                "   - Note any limitations or uncertainties\n"
                "   - Use data patterns to inform conclusions\n\n"
                "   - Data Analysis and code execution tools are available to assist you\n\n"
                "   - If you use the 'execute_code' tool, show the code you used as evidence and always store the final result of the code in a variable 'result'. Returning these variables will be handled for you will be handled for you\n\n"
                "   - Any code written should be restricted to data analysis use cases. Avoid code that might be insecure or lead to side effects\n\n"
                
                "Example of correctly formatted response:\n"
                '{"answer": "# Analysis Results 🔍\\n\\n'
                '**Key Finding:** The data shows interesting patterns\\n\\n'
                '## Details\\n\\n'
                '- The `column_name` shows X\\n'
                '- Statistics indicate Y\\n\\n'
                '``` Evidence: Z```"}'

                "Example of code execution tool usage:\n"
                'result = df["Age"].sum()\n'

                + (f"\n\nPrevious Conversation Context:\n{context_text}" if context_text else "")
            )


            # First, get the model to choose appropriate tools
            tool_selection_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
                temperature=self.temperature
            )

            tool_results = []
            
            if tool_selection_response.choices[0].message.tool_calls:
                for tool_call in tool_selection_response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "execute_code":
                        code = function_args.get("code", "")
                        result = self.tools.execute_code(code) 
                        
                        print("Code Execution Result:", result)
                        
                        tool_result = {
                            "tool": function_name,
                            "args": function_args,
                            "result": result
                        }
                        
                        tool_results.append(tool_result)
                    
                    # Handle other tools
                    elif hasattr(self.tools, function_name):
                        result = getattr(self.tools, function_name)(**function_args)
                        tool_results.append({
                            "tool": function_name,
                            "args": function_args,
                            "result": result
                        })

            # Generate final response using tool results
            final_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            if tool_results:
                # Format tool results for better readability
                formatted_results = self._format_tool_results(tool_results)
                final_messages.append({
                    "role": "assistant",
                    "content": f"Analysis Results:\n\n{formatted_results}"
                })

            final_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=final_messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            if not final_response or not final_response.choices:
                 raise APIError("Invalid or empty response from the API.")

            response_text = final_response.choices[0].message.content.strip()

            try:
                # Clean up the response if it contains markdown-style code blocks
                response_text = response_text.replace("```json", "").replace("```", "").strip()
                parsed_response = extract_json(response_text)
                
                if parsed_response is None or 'answer' not in parsed_response:
                    # If JSON parsing fails or 'answer' key is missing, return the raw response with formatting
                    formatted_response = (
                        "## ⚠️ Response Format Note\n\n"
                        "I received a response but it wasn't in the expected JSON format. "
                        "Here's the raw response:\n\n"
                        "---\n\n"
                        f"{response_text}\n\n"
                        "---\n\n"
                        "_Please try asking your question again._"
                    )
                    return formatted_response
                
                answer = parsed_response['answer']

                # Store conversation history if memory is enabled
                if self.enable_memory:
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

            except Exception as json_error:
                # Handle JSON parsing errors with a user-friendly message
                formatted_response = (
                    "## ⚠️ Response Processing Error\n\n"
                    "I encountered an error while processing the response. "
                    f"Error details: {str(json_error)}\n\n"
                    "Here's the raw response I received:\n\n"
                    "---\n\n"
                    f"{response_text}\n\n"
                    "---\n\n"
                    "_Please try rephrasing your question._"
                )
                return formatted_response

        except Exception as e:
            logger.error(f"Error: {e}")
            raise APIError(f"Chat request failed: {str(e)}")

    def __call__(self, query: str) -> str:
        """
        Ask a question about the DataFrame.
        
        Args:
            query (str): The question about the DataFrame
            
        Returns:
            str: The response from the model
        """
        return self._get_chat_response(query)

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

