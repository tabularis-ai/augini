import logging
from datetime import datetime
from openai import OpenAI
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .exceptions import APIError
from .utils import extract_json
import ipywidgets as widgets
from IPython.display import display, HTML

class AuginiInteractiveChat:
    def __init__(self, augini_instance, dataframe: pd.DataFrame):
        """
        Initialize the interactive chat interface
        
        Args:
            augini_instance: An instance of the Augini class
            dataframe: The DataFrame to analyze
        """
        self.augini = augini_instance
        self.df = dataframe
        self.chat_history: List[Dict] = []
        
        # Create widgets
        self.text_input = widgets.Text(
            placeholder='Type your question here...',
            description='Query:',
            layout=widgets.Layout(width='80%')
        )
        
        self.send_button = widgets.Button(
            description='Send',
            button_style='primary',
            icon='paper-plane'
        )
        
        self.clear_button = widgets.Button(
            description='Clear Chat',
            button_style='warning',
            icon='trash'
        )
        
        self.export_button = widgets.Button(
            description='Export Chat',
            button_style='info',
            icon='download'
        )
        
        self.exit_button = widgets.Button(
            description='Exit Chat',
            button_style='danger',
            icon='sign-out'
        )
        
        self.chat_output = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='10px',
                margin='10px 0',
                max_height='400px',
                overflow_y='auto'
            )
        )
        
        # Create chat container
        self.chat_container = widgets.VBox([
            widgets.HBox([self.text_input, self.send_button]),
            widgets.HBox([self.clear_button, self.export_button, self.exit_button]),
            self.chat_output
        ])
        
        # Attach event handlers
        self.send_button.on_click(self._on_send)
        self.clear_button.on_click(self._on_clear)
        self.export_button.on_click(self._on_export)
        self.exit_button.on_click(self._on_exit)
        self.text_input.on_submit(self._on_send)
        
        # Display initial message
        with self.chat_output:
            display(HTML(
                "<div style='color: #666; padding: 10px;'>"
                "👋 Welcome to Augini Interactive Chat! Ask me anything about your data."
                "<br>Type your question and press Enter or click Send."
                "<br>Click 'Clear Chat' to reset the conversation or 'Exit Chat' to end."
                "</div>"
            ))
    
    def _format_message(self, content: str, is_user: bool = False) -> str:
        """Format chat messages with appropriate styling"""
        style = """
            padding: 10px;
            margin: 5px;
            border-radius: 10px;
            max-width: 80%;
            {}
        """.format(
            "background-color: #e3f2fd; margin-left: auto;" if is_user 
            else "background-color: #f5f5f5; margin-right: auto;"
        )
        
        return f"<div style='{style}'>{content}</div>"
    
    def _add_to_history(self, role: str, content: str):
        """Add a message to chat history with timestamp"""
        self.chat_history.append({
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content
        })
    
    def _on_send(self, _):
        """Handle send button click or input submission"""
        query = self.text_input.value.strip()
        if not query:
            return
            
        # Clear input
        self.text_input.value = ''
        
        # Display user message and add to history
        with self.chat_output:
            display(HTML(self._format_message(f"<b>You:</b> {query}", is_user=True)))
        self._add_to_history('user', query)
        
        try:
            # Get response from Augini (without memory)
            response = self.augini.chat(query, self.df, use_memory=False)
            
            # Display assistant message and add to history
            with self.chat_output:
                display(HTML(self._format_message(f"<b>Assistant:</b> {response}")))
            self._add_to_history('assistant', response)
            
        except Exception as e:
            error_msg = f"<span style='color: red'>Error: {str(e)}</span>"
            with self.chat_output:
                display(HTML(self._format_message(error_msg)))
            self._add_to_history('system', f"Error: {str(e)}")
    
    def _on_clear(self, _):
        """Handle clear button click"""
        self.chat_history = []
        self.chat_output.clear_output()
        
        # Display welcome message again
        with self.chat_output:
            display(HTML(
                "<div style='color: #666; padding: 10px;'>"
                "Chat cleared! Ask me a new question about your data."
                "</div>"
            ))
    
    def _on_export(self, _):
        """Handle export button click - Export chat history to markdown"""
        if not self.chat_history:
            with self.chat_output:
                display(HTML(
                    "<div style='color: orange; padding: 10px;'>"
                    "No chat history to export!"
                    "</div>"
                ))
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.md"
        
        with open(filename, 'w') as f:
            f.write("# Augini Chat History\n\n")
            for msg in self.chat_history:
                time = datetime.fromisoformat(msg['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                role = msg['role'].title()
                content = msg['content']
                f.write(f"## {role} - {time}\n\n{content}\n\n---\n\n")
        
        with self.chat_output:
            display(HTML(
                f"<div style='color: green; padding: 10px;'>"
                f"Chat history exported to {filename}"
                f"</div>"
            ))
    
    def _on_exit(self, _):
        """Handle exit button click"""
        self.chat_output.clear_output()
        with self.chat_output:
            display(HTML(
                "<div style='color: #666; padding: 10px;'>"
                "👋 Chat session ended. Thank you for using Augini Interactive Chat!"
                "</div>"
            ))
        
    def start(self):
        """Start the interactive chat session"""
        display(self.chat_container)
        
    def get_chat_history(self) -> List[Dict]:
        """Return the chat history"""
        return self.chat_history