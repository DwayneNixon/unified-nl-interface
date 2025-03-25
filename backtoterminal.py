import streamlit as st
import speech_recognition as sr
import logging
from typing import Dict, List, Any
from interpreter import interpreter
import traceback
import subprocess
import os

# Flexible translator import with robust error handling
def get_translator():
    """
    Attempt to import Translator with multiple fallback strategies.
    
    Returns:
        A translation object with translate method
    """
    import_attempts = [
        # Try standard googletrans import
        lambda: __import__('googletrans').Translator(),
        
        # Try deep translator as alternative
        lambda: __import__('deep_translator').GoogleTranslator(),
        
        # Fallback custom translation wrapper
        lambda: type('FallbackTranslator', (), {
            'translate': lambda self, text, dest='en': type('TranslationResult', (), {
                'text': text,
                'src': 'auto'
            })()
        })()
    ]

    for attempt in import_attempts:
        try:
            return attempt()
        except (ImportError, Exception):
            continue
    
    raise ImportError("Could not import a translation library")

class MultilingualNLInterface:
    def __init__(self):
        """
        Initialize the Multilingual Natural Language Interface.
        """
        # Logging configuration
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Translator initialization
        self._initialize_translator()

        # Interpreter configuration
        self._configure_interpreter()

        # Language mappings
        self._setup_language_mappings()

        # Session state initialization
        self._initialize_session_state()

    def _initialize_translator(self):
        """
        Safely initialize the translator with error handling.
        """
        try:
            self.translator = get_translator()
            self.logger.info("Translation library successfully imported")
        except Exception as e:
            self.logger.error(f"Translator initialization failed: {e}")
            st.error(f"Translation service initialization error: {e}")
            self.translator = None

    def _configure_interpreter(self):
        """
        Configure Open Interpreter for local command execution.
        """
        try:
            # Reset and configure interpreter for local execution
            interpreter.reset()
            interpreter.auto_run = True  # Auto-approve commands
            interpreter.llm.model = "ollama/llama3.1"  # Optional: specify local LLM
            interpreter.llm.api_base = "http://localhost:11434"
            interpreter.verbose = True
            
            # Critical: Enable local command execution
            interpreter.system_message = """
            You are an AI that can execute local shell commands. 
            Always provide a clear explanation before running any command.
            Be extremely careful and only run safe, non-destructive commands.
            Never run commands that could harm the system or compromise security.
            """
            
            self.logger.info("Open Interpreter configured for local execution")
        except Exception as e:
            self.logger.error(f"Interpreter configuration error: {e}")
            st.error("Failed to configure interpreter. Check settings.")

    def _setup_language_mappings(self):
        """
        Define language codes and mappings.
        """
        self.language_codes = {
            "English": "en", "Hindi": "hi", "Marathi": "mr", 
            "Bengali": "bn", "Tamil": "ta", "Telugu": "te", 
            "Kannada": "kn", "Malayalam": "ml", "Gujarati": "gu", 
            "Punjabi": "pa", "Spanish": "es", "French": "fr", 
            "German": "de", "Arabic": "ar", "Chinese": "zh-cn"
        }
        self.language_names = {v: k for k, v in self.language_codes.items()}

    def _initialize_session_state(self):
        """
        Safely initialize Streamlit session state variables.
        """
        default_states = {
            "voice_message": "",
            "translated_message": "",
            "original_language": "",
            "chat_history": [],
            "error_count": 0,
            "selected_language": "English",
            "current_directory": os.getcwd()
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def translate_text(self, text: str, target_lang: str = 'en') -> Dict[str, Any]:
        """
        Robust text translation with detailed error handling.
        """
        if not self.translator:
            st.warning("Translation service unavailable")
            return {
                "original_text": text,
                "translated_text": text,
                "original_language": "Unknown",
                "confidence": 0
            }

        try:
            # Language detection
            original_lang_code = getattr(self.translator.detect(text), 'lang', 'unknown')
            original_lang = self.language_names.get(original_lang_code, original_lang_code)
            
            # Translation
            translated_text = self.translator.translate(text, dest=target_lang).text
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "original_language": original_lang,
                "confidence": 1.0
            }
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            st.error(f"Translation failed: {e}")
            return {
                "original_text": text,
                "translated_text": text,
                "original_language": "Unknown",
                "confidence": 0
            }

    def render_content(self, grouped_chunks: Dict[str, List[str]]):
        """
        Advanced content rendering with type-specific formatting.
        """
        content_renderers = {
            "code": lambda content: st.code(content, language="python"),
            "console": lambda content: st.text_area(
                "Console Output:", 
                value=content, 
                height=200, 
                disabled=True
            ),
            "confirmation": st.success,
            "message": st.markdown
        }

        for chunk_type, content_list in grouped_chunks.items():
            if not content_list:
                continue

            renderer = content_renderers.get(
                chunk_type, 
                lambda content: st.warning(f"Unsupported content type: {content}")
            )

            combined_content = "\n".join(content_list).strip()
            renderer(combined_content)

    def run(self):
        """
        Main application runner with enhanced UI and interaction.
        """
        st.set_page_config(
            page_title="Multilingual Command Interface", 
            page_icon="🌐",
            layout="wide"
        )
        
        st.title("🌐 Multilingual Local Command Interface")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("🔧 System Configuration")
            st.json({
                "Current Directory": st.session_state.current_directory,
                "Interpreter Mode": "Local Command Execution",
                "Auto Run": "Enabled"
            })
            
            st.divider()
            st.metric("Total Errors", st.session_state.error_count)

        # Language and input columns
        col_lang, col_input = st.columns([1, 3])
        
        with col_lang:
            selected_language = st.selectbox(
                "Select Language", 
                list(self.language_codes.keys()),
                index=list(self.language_codes.keys()).index(st.session_state.selected_language)
            )
            st.session_state.selected_language = selected_language

        with col_input:
            user_input = st.text_input(
                f"Enter your command in {selected_language}", 
                value=st.session_state.voice_message, 
                placeholder=f"Type a local command or query in {selected_language}..."
            )
        
        # Message processing
        if st.button("Execute Command", type="primary"):
            if user_input.strip():
                try:
                    with st.spinner("Processing command..."):
                        # Translation
                        translation_result = self.translate_text(user_input)
                        translated_message = translation_result['translated_text']
                        original_language = translation_result['original_language']

                        # Display translation info
                        st.info(f"Original Language: {original_language}")
                        st.info(f"Translated Command: {translated_message}")

                        # Process with interpreter (local command execution)
                        response_stream = interpreter.chat(translated_message, stream=True)
                        
                        grouped_chunks = {
                            "code": [], "console": [], 
                            "confirmation": [], "message": []
                        }
                        
                        for chunk in response_stream:
                            chunk_type = chunk.get("type", "message")
                            content = chunk.get("content", "")
                            
                            if isinstance(content, str) and content.strip():
                                grouped_chunks.setdefault(chunk_type, []).append(content)
                        
                        # Render response
                        self.render_content(grouped_chunks)
                        
                        # Update chat history
                        st.session_state.chat_history.append({
                            "user": user_input,
                            "translated_user": translated_message,
                            "language": selected_language,
                            "original_language": original_language,
                            "response": grouped_chunks
                        })
                
                except Exception as e:
                    st.error(f"Command processing error: {e}")
                    self.logger.error(traceback.format_exc())
                    st.session_state.error_count += 1
            else:
                st.warning("Please enter a command before executing.")

def main():
    interface = MultilingualNLInterface()
    interface.run()

if __name__ == "__main__":
    main()