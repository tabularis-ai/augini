"""Web components for interactive chat interface."""

import ipywidgets as widgets
from IPython.display import display, HTML, Javascript
import markdown

class ChatComponents:
    def __init__(self):
        """Initialize chat interface components."""
        try:
            self.setup_components()
        except ImportError:
            raise ImportError(
                "Interactive features require ipywidgets. "
                "Install with: pip install ipywidgets"
            )

    def setup_components(self):
        """Set up all chat interface components."""
        # Create components first
        self.components = {
            'chat_output': self._create_chat_output(),
            'chat_history': [],
            'loading': self._create_loading_indicator(),
            **self._create_buttons_and_input()
        }

        # Create layout containers
        input_area = self._create_input_area()
        button_toolbar = self._create_button_toolbar()
        self.main_container = self._create_main_container(input_area, button_toolbar)
        self.components['container'] = self.main_container

    def _create_chat_output(self):
        """Create the chat output area."""
        output = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                height='500px',  # Increased height for better visibility
                overflow_y='scroll',  # Changed to scroll to ensure scrollbar is always visible
                padding='15px',
                margin='0 0 60px 0',  # Added bottom margin for spacing
                flex='1 1 auto'
            )
        )
        return output

    def _create_loading_indicator(self):
        """Create the loading indicator."""
        return widgets.HTML(
            value=(
                "<div style='text-align: center; padding: 10px;'>"
                "<i class='fa fa-circle-o-notch fa-spin' style='color: #007bff;'></i>"
                "</div>"
            ),
            layout=widgets.Layout(visibility='hidden')
        )

    def _create_buttons_and_input(self):
        """Create buttons and input field."""
        return {
            'text_input': widgets.Text(
                placeholder='Type your question here...',
                layout=widgets.Layout(
                    flex='1',
                    height='36px'
                )
            ),
            'send_button': widgets.Button(
                description='Send',
                button_style='primary',
                icon='paper-plane',
                layout=widgets.Layout(
                    width='70px',
                    height='36px'
                )
            ),
            'clear_button': widgets.Button(
                description='Clear',
                button_style='warning',
                icon='trash',
                layout=widgets.Layout(
                    width='70px',
                    height='32px'
                )
            )
        }

    def _create_input_area(self):
        """Create the input area container."""
        return widgets.HBox(
            [self.components['text_input'], 
             self.components['send_button']],
            layout=widgets.Layout(
                width='100%',
                padding='10px 15px',
                background_color='white',
                border_top='1px solid #e0e0e0'
            )
        )

    def _create_button_toolbar(self):
        """Create the button toolbar."""
        return widgets.HBox(
            [self.components['clear_button']],  # Only clear button
            layout=widgets.Layout(
                width='100%',
                justify_content='flex-end',
                padding='10px 15px',
                background_color='#f8f9fa',
                border_top='1px solid #e0e0e0'
            )
        )

    def _create_main_container(self, input_area, button_toolbar):
        """Create the main container."""
        # Create a container for the fixed bottom controls
        bottom_controls = widgets.VBox(
            [button_toolbar, input_area],
            layout=widgets.Layout(
                width='100%',
                background_color='white',
                border_top='1px solid #e0e0e0',
                position='absolute',
                bottom='0',
                left='0',
                right='0'
            )
        )

        # Create a container for the chat content
        chat_content = widgets.VBox(
            [
                self.components['loading'],
                self.components['chat_output']
            ],
            layout=widgets.Layout(
                width='100%',
                height='calc(100% - 120px)',  # Adjust height to account for bottom controls
                overflow_y='hidden'  # Hide overflow on container
            )
        )

        # Create the main container with proper layout
        return widgets.VBox(
            [chat_content, bottom_controls],
            layout=widgets.Layout(
                width='100%',
                max_width='800px',
                height='650px',  # Increased height
                margin='0 auto',
                border='1px solid #e0e0e0',
                border_radius='8px',
                background_color='white',
                box_shadow='0 2px 6px rgba(0,0,0,0.1)',
                display='flex',
                flex_flow='column',
                overflow='hidden',
                position='relative'
            )
        )

    def format_message(self, content: str, is_user: bool = False) -> str:
        """Format chat messages with improved styling and markdown support."""
        if not is_user:
            try:
                content = markdown.markdown(content)
            except:
                pass

        message_styles = {
            'base': """
                padding: 12px 16px;
                margin: 8px 0;
                border-radius: 15px;
                max-width: 80%;
                line-height: 1.4;
                font-size: 14px;
                word-wrap: break-word;
                overflow-wrap: break-word;
            """,
            'user': """
                background-color: #007bff;
                color: white;
                margin-left: auto;
                margin-right: 0;
                border-bottom-right-radius: 5px;
            """,
            'assistant': """
                background-color: #f1f3f4;
                color: #202124;
                margin-right: auto;
                margin-left: 0;
                border-bottom-left-radius: 5px;
            """
        }

        container_style = message_styles['base'] + (
            message_styles['user'] if is_user else message_styles['assistant']
        )

        header_style = """
            font-size: 12px;
            font-weight: 500;
            margin-bottom: 4px;
            opacity: 0.7;
            color: inherit;
        """

        return f"""
            <div style='display: flex; margin: 8px 0;'>
                <div style='{container_style}'>
                    <div style='{header_style}'>{is_user and "You" or "Augini"}</div>
                    <div>{content}</div>
                </div>
            </div>
        """

    def scroll_to_bottom(self):
        """Scroll the chat output to the bottom."""
        script = """
            setTimeout(function() {
                var output = document.querySelector('.jp-OutputArea-output');
                if (output) {
                    output.scrollTop = output.scrollHeight;
                }
            }, 100);
        """
        display(Javascript(script))

    def display_welcome_message(self):
        """Display the welcome message in the chat."""
        with self.components['chat_output']:
            display(HTML(
                "<div style='text-align: center; padding: 20px; color: #666;'>"
                "<h3 style='margin: 0 0 10px 0;'>👋 Welcome to Interactive Chat</h3>"
                "<p style='margin: 0;'>Ask me anything about your data.</p>"
                "</div>"
            ))

    def display(self):
        """Display the chat interface."""
        display(self.main_container) 