import streamlit as st
import traceback
import uuid
import logging
from datetime import datetime
import yaml
from dataclasses import dataclass, asdict

# Import necessary libraries
import speech_recognition as sr
from interpreter import interpreter
import html
import bleach
import re
from typing import List, Optional, Dict

class ConfigManager:
    """
    Manages application configuration
    """
    @staticmethod
    def load_config(config_path='config.yaml'):
        """
        Load configuration from YAML file
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return {
                'model': 'ollama/llama3.1',
                'api_base': 'http://localhost:11434',
                'max_tokens': 4096,
                'temperature': 0.7
            }

class InputValidator:
    """
    Input validation and sanitization
    """
    @staticmethod
    def sanitize_input(input_text: str) -> str:
        """
        Sanitize input to prevent XSS and injection
        """
        # HTML escape
        escaped_text = html.escape(input_text)
        
        # Use bleach to strip potentially harmful HTML
        cleaned_text = bleach.clean(
            escaped_text, 
            tags=[], 
            strip=True
        )
        
        return cleaned_text
    
    @staticmethod
    def validate_input(
        input_text: str, 
        max_length: int = 1000, 
        forbidden_patterns: List[str] = None
    ) -> bool:
        """
        Validate input against various criteria
        """
        if not input_text:
            return False
        
        # Length check
        if len(input_text) > max_length:
            return False
        
        # Default forbidden patterns
        default_forbidden = [
            r'rm\s+-rf',  # Dangerous shell command
            r'sudo\s+',   # Prevent sudo usage
            r'wget\s+',   # Prevent downloads
            r'curl\s+',   # Prevent network requests
        ]
        
        forbidden_patterns = forbidden_patterns or default_forbidden
        
        # Check forbidden patterns
        for pattern in forbidden_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                return False
        
        return True

class ErrorLogger:
    """
    Comprehensive error logging system
    """
    def __init__(self, log_dir='logs'):
        import os
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        log_filename = f"{log_dir}/interpreter_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_error(self, error, context=None):
        """
        Log detailed error information
        """
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.logger.error(f"Error Details: {error_details}")
        return error_details

class RateLimiter:
    """
    Track and limit request frequency
    """
    def __init__(self, max_requests_per_minute=10):
        from collections import defaultdict
        self.request_log = defaultdict(list)
        self.max_requests_per_minute = max_requests_per_minute
    
    def can_make_request(self, user_id):
        """
        Check if user can make a request based on rate limits
        """
        from datetime import datetime, timedelta
        
        current_time = datetime.now()
        one_minute_ago = current_time - timedelta(minutes=1)
        
        # Remove old request timestamps
        self.request_log[user_id] = [
            timestamp for timestamp in self.request_log[user_id] 
            if timestamp > one_minute_ago
        ]
        
        # Check if within rate limit
        if len(self.request_log[user_id]) >= self.max_requests_per_minute:
            return False
        
        # Log current request
        self.request_log[user_id].append(current_time)
        return True

class ConversationManager:
    """
    Manage conversation history
    """
    def __init__(self, max_history_length=50):
        self.conversations = {}
        self.max_history_length = max_history_length
    
    def add_conversation(self, session_id, user_message, system_response):
        """
        Add a conversation entry
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        conversation_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'user_message': user_message,
            'system_response': system_response
        }
        
        self.conversations[session_id].append(conversation_entry)
        
        # Maintain max history length
        if len(self.conversations[session_id]) > self.max_history_length:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history_length:]
    
    def get_conversation_context(self, session_id, num_recent_entries=5):
        """
        Get recent conversation context
        """
        if session_id not in self.conversations:
            return []
        
        return self.conversations[session_id][-num_recent_entries:]

class VoiceInputHandler:
    """
    Handle voice input using speech recognition
    """
    @staticmethod
    def get_voice_input():
        """
        Capture voice input
        """
        recognizer = sr.Recognizer()
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
                st.info("🎤 Listening... Speak clearly into the microphone.")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            # Transcribe audio
            voice_message = recognizer.recognize_google(audio)
            st.success(f"✅ Captured: {voice_message}")
            return voice_message
        
        except sr.WaitTimeoutError:
            st.error("⏰ Listening timed out. Please try again.")
        except sr.UnknownValueError:
            st.error("🤷 Could not understand the audio. Please speak more clearly.")
        except sr.RequestError as e:
            st.error(f"❌ Speech recognition service error: {e}")
        
        return ""

class OpenInterpreterApp:
    def __init__(self):
        # Initialize components
        self.config = ConfigManager.load_config()
        self.error_logger = ErrorLogger()
        self.rate_limiter = RateLimiter()
        self.conversation_manager = ConversationManager()
        
        # Configure Open Interpreter
        interpreter.offline = True
        interpreter.llm.model = self.config.get('model', 'ollama/llama3.1')
        interpreter.llm.api_base = self.config.get('api_base', 'http://localhost:11434')
        
        # Initialize Streamlit page
        st.set_page_config(
            page_title="Unified NL Interpreter", 
            page_icon=":robot_face:", 
            layout="wide"
        )
    
    def process_message(self, message, session_id):
        """
        Process user message with comprehensive checks
        """
        # Sanitize input
        sanitized_message = InputValidator.sanitize_input(message)
        
        # Validate input
        if not InputValidator.validate_input(sanitized_message):
            st.error("Invalid input. Please check your message.")
            return None
        
        # Rate limit check
        if not self.rate_limiter.can_make_request(session_id):
            st.error("Rate limit exceeded. Please wait a moment.")
            return None
        
        try:
            # Get conversation context
            context = self.conversation_manager.get_conversation_context(session_id)
            
            # Process message with Open Interpreter
            with st.spinner("Processing your request..."):
                response_stream = interpreter.chat(sanitized_message, stream=True)
                
                # Collect response chunks
                grouped_chunks = {
                    "code": [],
                    "console": [],
                    "confirmation": [],
                    "message": [],
                }
                
                for chunk in response_stream:
                    chunk_type = chunk.get("type", "message")
                    content = chunk.get("content", "")
                    
                    if isinstance(content, str) and content.strip():
                        grouped_chunks.setdefault(chunk_type, []).append(content)
                
                # Combine response
                full_response = "\n".join(
                    "\n".join(contents) 
                    for contents in grouped_chunks.values() 
                    if contents
                )
                
                # Add to conversation history
                self.conversation_manager.add_conversation(
                    session_id, sanitized_message, full_response
                )
                
                return grouped_chunks
        
        except Exception as e:
            # Log and handle errors
            error_context = {
                'user_message': sanitized_message,
                'timestamp': datetime.now()
            }
            error_details = self.error_logger.log_error(e, context=error_context)
            st.error(f"An error occurred: {e}")
            return None
    
    def display_grouped_content(self, grouped_chunks):
        """
        Display content from different response chunks
        """
        content_displays = {
            "code": lambda content: st.code("".join(content).strip(), language="python"),
            "console": lambda content: st.text_area("Console Output:", "\n".join(content).strip(), height=200, disabled=True),
            "confirmation": lambda content: st.success("\n".join(content).strip()),
            "message": lambda content: st.markdown("\n".join(content).strip()),
        }
        
        for chunk_type, contents in grouped_chunks.items():
            if contents:
                display_func = content_displays.get(chunk_type, lambda x: st.warning(f"Unknown type: {chunk_type}\n{x}"))
                display_func(contents)
    
    # def run(self):
    #     """
    #     Main application runner
    #     """
    #     # Generate unique session ID
    #     if 'session_id' not in st.session_state:
    #         st.session_state.session_id = str(uuid.uuid4())
        
    #     st.title("🤖 Unified Natural Language Interpreter")
        
    #     # Sidebar for configuration
    #     with st.sidebar:
    #         st.header("🛠️ Configuration")
    #         st.json(self.config)
        
    #     # Input columns
    #     col1, col2 = st.columns([3, 1])
        
    #     with col1:
    #         message = st.text_input(
    #             "Enter your message:", 
    #             key="message_input",
    #             placeholder="Type a command or describe a task..."
    #         )
        
    #     with col2:
    #         st.write("") # Spacer
    #         voice_input_button = st.button("🎤 Voice Input")
        
    #     # Voice input handling
    #     if voice_input_button:
    #         voice_message = VoiceInputHandler.get_voice_input()
    #         if voice_message:
    #             # Update text input with voice message
    #             st.session_state.message_input = voice_message
    #             st.experimental_rerun()
        
    #     # Send message button
    #     if st.button("🚀 Send Message", use_container_width=True):
    #         if message.strip():
    #             # Process message
    #             response = self.process_message(
    #                 message, 
    #                 st.session_state.session_id
    #             )
                
    #             # Display response
    #             if response:
    #                 self.display_grouped_content(response)



    def run(self):
        """
        Main application runner
        """
        # Generate unique session ID
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        # Initialize session state for message input if not exists
        if 'message_input' not in st.session_state:
            st.session_state.message_input = ""
        
        st.title("🤖 Unified Natural Language Interpreter")
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("🛠️ Configuration")
            st.json(self.config)
        
        # Input columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Use session state value for initial input
            message = st.text_input(
                "Enter your message:", 
                value=st.session_state.message_input,
                key="message_input",
                placeholder="Type a command or describe a task..."
            )
        
        with col2:
            st.write("") # Spacer
            voice_input_button = st.button("🎤 Voice Input")
        
        # Voice input handling
        if voice_input_button:
            voice_message = VoiceInputHandler.get_voice_input()
            if voice_message:
                # Update session state instead of trying to modify widget directly
                st.session_state.message_input = voice_message
        
        # Send message button
        if st.button("🚀 Send Message", use_container_width=True):
            if message.strip():
                # Process message
                response = self.process_message(
                    message, 
                    st.session_state.session_id
                )
                
                # Display response
                if response:
                    self.display_grouped_content(response)
                
                # Clear the input after processing
                st.session_state.message_input = ""

def main():
    app = OpenInterpreterApp()
    app.run()

if __name__ == "__main__":
    main()