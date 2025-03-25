import streamlit as st
import speech_recognition as sr
import logging
from typing import Dict, List, Any
from interpreter import interpreter
import traceback
import os
import ast
import textwrap

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

class MultilingualCommandSuggestionInterface:
    def __init__(self):
        """
        Initialize the Multilingual Command Suggestion Interface.
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
        Configure Open Interpreter for command suggestion mode.
        """
        try:
            # Reset and configure interpreter
            interpreter.reset()
            
            # Configuration for command suggestion
            interpreter.auto_run = False  # Disable auto-execution
            interpreter.llm.model = "ollama/llama3.1"
            interpreter.llm.api_base = "http://localhost:11434"
            interpreter.verbose = True
            
            # System message for generating safe, explainable commands
            interpreter.system_message = """
            You are an AI assistant that generates Python command suggestions.
            Your tasks:
            1. Generate safe, clear Python commands
            2. Focus exclusively on Python solutions
            3. Provide detailed comments explaining each command
            4. Break down complex tasks into simple, executable Python code
            5. Include error handling and type hints where appropriate
            6. Suggest multiple Pythonic approaches if applicable
            """
            
            self.logger.info("Open Interpreter configured for command suggestion")
        except Exception as e:
            self.logger.error(f"Interpreter configuration error: {e}")
            st.error("Failed to configure interpreter. Check Ollama and Llama model.")

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
            "current_directory": os.getcwd(),
            "suggested_commands": None,
            "command_explanation": ""
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """
        Safely execute Python code and capture output or errors.
        
        Args:
            code (str): Python code to execute
        
        Returns:
            Dict containing execution results
        """
        # Capture stdout and stderr
        from io import StringIO
        import sys

        # Result dictionary
        result = {
            "output": "",
            "error": "",
            "success": False
        }

        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            # Compile and execute the code
            compiled_code = compile(code, '<string>', 'exec')
            exec(compiled_code, {})
            
            # Capture output
            result["output"] = stdout_capture.getvalue().strip()
            result["success"] = True
        
        except Exception as e:
            # Capture error
            result["error"] = str(e)
            result["success"] = False
        
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return result

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

    def get_voice_input(self, language: str) -> str:
        """
        Capture voice input with enhanced error handling.
        
        Args:
            language (str): Language for voice recognition
        
        Returns:
            str: Transcribed voice message
        """
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        language_code = self.language_codes.get(language, "en-US")

        with sr.Microphone() as source:
            st.info(f"Listening in {language}... Speak clearly")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                voice_text = recognizer.recognize_google(audio, language=language_code)
                self.logger.info(f"Voice input successfully transcribed in {language}")
                return voice_text
            
            except sr.WaitTimeoutError:
                st.warning("Listening timed out. Please try again.")
            except sr.UnknownValueError:
                st.error(f"Could not understand audio in {language}. Please speak clearly.")
            except sr.RequestError as e:
                st.error(f"Speech recognition service error: {e}")
                st.session_state.error_count += 1
            
            return ""

    def generate_command_suggestions(self, translated_message: str) -> Dict[str, Any]:
        """
        Generate Python command suggestions using Open Interpreter.
        
        Args:
            translated_message (str): Translated user input
        
        Returns:
            Dict containing suggested commands and explanations
        """
        try:
            # Use interpreter to generate Python-specific suggestions
            response_stream = interpreter.chat(
                f"Generate ONLY Python code for the following task: {translated_message}. "
                "Provide multiple Python solutions with detailed comments. "
                "Ensure all code is syntactically correct and executable Python.",
                stream=True
            )
            
            suggestions = {
                "commands": [],
                "explanation": ""
            }
            
            current_section = "explanation"
            
            for chunk in response_stream:
                chunk_type = chunk.get("type", "message")
                content = chunk.get("content", "")
                
                # Focus on extracting Python code blocks
                if chunk_type == "code" and content.strip().startswith("```python"):
                    # Remove markdown code block markers
                    python_code = content.replace("```python", "").replace("```", "").strip()
                    suggestions["commands"].append(python_code)
                    current_section = "commands"
                elif chunk_type == "message" and content.strip():
                    if current_section == "explanation":
                        suggestions["explanation"] += content + "\n"
                    else:
                        # Attempt to extract code from text if it looks like Python
                        try:
                            # Remove any code block markers
                            cleaned_content = content.replace("```python", "").replace("```", "")
                            # Validate it's valid Python syntax
                            ast.parse(cleaned_content)
                            suggestions["commands"].append(cleaned_content)
                        except SyntaxError:
                            # If not valid Python, treat as explanation
                            suggestions["explanation"] += content + "\n"
            
            return suggestions
        
        except Exception as e:
            st.error(f"Command suggestion generation error: {e}")
            self.logger.error(traceback.format_exc())
            return {"commands": [], "explanation": "Failed to generate suggestions."}

    def run(self):
        """
        Main application runner with enhanced UI and improved formatting.
        """
        st.set_page_config(
            page_title="Multilingual Command Suggestion Interface", 
            page_icon="🌐",
            layout="wide"
        )
        
        st.title("🌐 Multilingual Command Suggestion Interface")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("🔧 System Configuration")
            st.json({
                "Current Directory": st.session_state.current_directory,
                "Model": "Ollama/Llama3.1",
                "Mode": "Command Suggestion"
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
                f"Enter your task in {selected_language}", 
                value=st.session_state.voice_message, 
                placeholder=f"Describe a task or problem in {selected_language}..."
            )
        
        # Voice input button
        voice_button = st.button("🎙️ Voice Input")

        if voice_button:
            voice_message = self.get_voice_input(selected_language)
            if voice_message:
                st.session_state.voice_message = voice_message
                st.rerun()

        # Suggestion generation
        if st.button("Generate Suggestions", type="primary"):
            if user_input.strip():
                try:
                    with st.spinner("Generating command suggestions..."):
                        # Translation
                        translation_result = self.translate_text(user_input)
                        translated_message = translation_result['translated_text']
                        original_language = translation_result['original_language']

                        # Create a more structured and readable output
                        st.header("🔍 Analysis Results")
                        
                        # Language Detection Card
                        with st.expander("🌍 Language Detection"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Detected Language", original_language)
                            with col2:
                                st.metric("Translation Confidence", 
                                          f"{translation_result.get('confidence', 0)*100:.0f}%")
                        
                        # Translation Card
                        st.subheader("📝 Translation")
                        translation_container = st.container()
                        with translation_container:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Original Text:**")
                                st.code(user_input, language=original_language)
                            with col2:
                                st.write("**Translated Text:**")
                                st.code(translated_message, language="en")

                        # Generate command suggestions
                        suggestions = self.generate_command_suggestions(translated_message)
                        
                        # Store suggestions in session state
                        st.session_state.suggested_commands = suggestions['commands']
                        st.session_state.command_explanation = suggestions['explanation']

                        # Detailed Explanation Section
                        st.subheader("📘 Task Analysis")
                        with st.expander("Detailed Explanation"):
                            st.write(suggestions['explanation'])

                        # Suggested Commands Section
                        st.subheader("💻 Suggested Commands")
                        for idx, command in enumerate(suggestions['commands'], 1):
                            with st.expander(f"Command {idx}"):
                                # Syntax highlighted code display
                                st.code(command, language="python")
                                
                                # Execute Python code button
                                execute_button = st.button(f"Execute Command {idx}", key=f"execute_{idx}")
                                if execute_button:
                                    # Execute the Python code
                                    execution_result = self.execute_python_code(command)
                                    
                                    # Display execution results
                                    if execution_result['success']:
                                        st.success("Code Executed Successfully!")
                                        if execution_result['output']:
                                            st.subheader("Output:")
                                            st.code(execution_result['output'], language="text")
                                    else:
                                        st.error("Execution Failed")
                                        st.code(execution_result['error'], language="text")
                        
                        # Update chat history
                        st.session_state.chat_history.append({
                            "user": user_input,
                            "translated_user": translated_message,
                            "language": selected_language,
                            "original_language": original_language,
                            "suggestions": suggestions
                        })
                
                except Exception as e:
                    st.error(f"Suggestion generation error: {e}")
                    self.logger.error(traceback.format_exc())
                    st.session_state.error_count += 1
            else:
                st.warning("Please enter a task before generating suggestions.")

def main():
    interface = MultilingualCommandSuggestionInterface()
    interface.run()

if __name__ == "__main__":
    main()