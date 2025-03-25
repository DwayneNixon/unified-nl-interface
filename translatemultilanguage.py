import streamlit as st
import speech_recognition as sr
import logging
from typing import Dict, List
from interpreter import interpreter
import sys
import traceback

# Flexible translator import with error handling
def get_translator():
    """
    Attempt to import Translator with multiple fallback strategies
    """
    import_attempts = [
        # Try standard googletrans import
        lambda: __import__('googletrans').Translator(),
        
        # Try deep translator as alternative
        lambda: __import__('deep_translator').GoogleTranslator(),
        
        # Try custom translation wrapper
        lambda: type('FallbackTranslator', (), {
            'translate': lambda self, text, dest='en': type('TranslationResult', (), {
                'text': text,
                'src': 'auto'
            })()
        })()
    ]

    last_error = None
    for attempt in import_attempts:
        try:
            return attempt()
        except (ImportError, Exception) as e:
            last_error = e
            continue
    
    # If all attempts fail, raise the last error
    raise ImportError(f"Could not import a translation library. Last error: {last_error}")

class MultilingualTranslatorInterface:
    def __init__(self):
        """
        Initialize the Multilingual Translation Natural Language Interface.
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize flexible translator
        try:
            self.translator = get_translator()
            self.logger.info("Translation library successfully imported")
        except Exception as e:
            self.logger.error(f"Translator initialization failed: {e}")
            st.error(f"Translation service initialization error: {e}")
            self.translator = None

        # Configure Open Interpreter
        self._configure_interpreter()

        # Configure language mappings
        self.language_codes = {
            "English": "en",
            "Hindi": "hi",
            "Marathi": "mr",
            "Bengali": "bn", 
            "Tamil": "ta",
            "Telugu": "te",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Gujarati": "gu",
            "Punjabi": "pa",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Arabic": "ar",
            "Chinese": "zh-cn"
        }

        # Reverse mapping for display
        self.language_names = {v: k for k, v in self.language_codes.items()}

        # Initialize session state
        self._initialize_session_state()

    def _configure_interpreter(self):
        """
        Configure the Open Interpreter settings.
        """
        try:
            interpreter.offline = True  # Disable online features
            interpreter.llm.model = "ollama/llama3.1"  # Use specified model
            interpreter.llm.api_base = "http://localhost:11434"  # Set API endpoint
            self.logger.info("Open Interpreter configured successfully")
        except Exception as e:
            self.logger.error(f"Interpreter configuration error: {e}")
            st.error("Failed to configure interpreter. Check settings.")

    def _initialize_session_state(self):
        """
        Initialize or reset session state variables.
        """
        session_vars = [
            "voice_message", 
            "translated_message",
            "original_language",
            "chat_history", 
            "error_count",
            "selected_language"
        ]
        for var in session_vars:
            if var not in st.session_state:
                if var in ["voice_message", "translated_message", "original_language"]:
                    st.session_state[var] = ""
                elif var == "chat_history":
                    st.session_state[var] = []
                elif var == "error_count":
                    st.session_state[var] = 0
                elif var == "selected_language":
                    st.session_state[var] = "English"

    def translate_text(self, text: str, target_lang: str = 'en') -> Dict[str, str]:
        """
        Translate text to English and detect original language.
        
        Args:
            text (str): Text to translate
            target_lang (str): Target language code (default: English)
        
        Returns:
            Dict containing original and translated text
        """
        if not self.translator:
            st.warning("Translation service is unavailable")
            return {
                "original_text": text,
                "translated_text": text,
                "original_language": "Unknown",
                "confidence": 0
            }

        try:
            # Detect language
            if hasattr(self.translator, 'detect'):
                detection = self.translator.detect(text)
                original_lang_code = detection.lang
                confidence = getattr(detection, 'confidence', 0)
            else:
                # Fallback for libraries without detect method
                original_lang_code = 'unknown'
                confidence = 0

            original_lang = self.language_names.get(original_lang_code, original_lang_code)

            # Translate to English
            if hasattr(self.translator, 'translate'):
                translation = self.translator.translate(text, dest=target_lang)
                translated_text = translation.text
            else:
                # Fallback for alternative translation libraries
                translated_text = self.translator.translate(text)
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "original_language": original_lang,
                "confidence": confidence
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
        Capture voice input using speech recognition in specified language.
        
        Args:
            language (str): Language for voice recognition
        
        Returns:
            str: Transcribed voice message
        """
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        
        # Get language code
        language_code = self.language_codes.get(language, "en-US")
        
        with sr.Microphone() as source:
            st.info(f"Listening in {language}... Speak into the microphone")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                st.success("Voice input captured. Processing...")
                
                # Use Google Speech Recognition with specific language
                voice_text = recognizer.recognize_google(audio, language=language_code)
                self.logger.info(f"Voice input successfully transcribed in {language}")
                return voice_text
            
            except sr.WaitTimeoutError:
                st.warning("Listening timed out. Please try again.")
            except sr.UnknownValueError:
                st.error(f"Could not understand the audio in {language}. Please speak clearly.")
            except sr.RequestError as e:
                st.error(f"Speech recognition service error: {e}")
                st.session_state.error_count += 1
            
            return ""

    def display_grouped_content(self, grouped_chunks: Dict[str, List[str]]):
        """
        Render different types of content chunks in Streamlit.
        
        Args:
            grouped_chunks (Dict[str, List[str]]): Categorized content chunks
        """
        content_type_map = {
            "code": (st.code, {"language": "python"}),
            "console": (st.text_area, {"label": "Console Output:", "height": 200, "disabled": True}),
            "confirmation": (st.success, {}),
            "message": (st.markdown, {})
        }

        for chunk_type, content_list in grouped_chunks.items():
            if not content_list:
                continue

            renderer, render_kwargs = content_type_map.get(
                chunk_type, 
                (st.warning, {"icon": "⚠️"})
            )

            combined_content = "\n".join(content_list).strip()
            
            if chunk_type == "code":
                renderer(combined_content, **render_kwargs)
            else:
                renderer(combined_content, **render_kwargs)
                
    def run(self):
        """
        Main application runner with multilingual support.
        """
        st.set_page_config(
            page_title="Multilingual NL Interface", 
            page_icon="🌐",
            layout="wide"
        )
        
        st.title("🚀 Multilingual Natural Language Interface")

        # Language Selection
        col_lang, col_input = st.columns([1, 3])
        
        with col_lang:
            selected_language = st.selectbox(
                "Select Language", 
                list(self.language_codes.keys()),
                index=list(self.language_codes.keys()).index(st.session_state.selected_language)
            )
            st.session_state.selected_language = selected_language

        # Main interaction area
        with col_input:
            message = st.text_input(
                "Enter your message:", 
                value=st.session_state.voice_message, 
                placeholder=f"Type your command or query in {selected_language}..."
            )
        
        # Voice input button
        voice_button = st.button("🎙️ Voice Input")

        if voice_button:
            voice_message = self.get_voice_input(selected_language)
            if voice_message:
                st.session_state.voice_message = voice_message
                st.rerun()

        # Message sending logic
        if st.button("Send Message", type="primary"):
            if message.strip():
                try:
                    with st.spinner("Processing your request..."):
                        # Translate the input to English
                        translation_result = self.translate_text(message)
                        translated_message = translation_result['translated_text']
                        original_language = translation_result['original_language']

                        # Display translation information
                        st.info(f"Original Language: {original_language}")
                        st.info(f"Translated Message: {translated_message}")

                        # Process translated message with interpreter
                        response_stream = interpreter.chat(translated_message, stream=True)
                        
                        grouped_chunks = {
                            "code": [],
                            "console": [],
                            "confirmation": [],
                            "message": []
                        }
                        
                        for chunk in response_stream:
                            chunk_type = chunk.get("type", "message")
                            content = chunk.get("content", "")
                            
                            if isinstance(content, str) and content.strip():
                                grouped_chunks.setdefault(chunk_type, []).append(content)
                        
                        self.display_grouped_content(grouped_chunks)
                        
                        # Optional: Update chat history
                        st.session_state.chat_history.append({
                            "user": message,
                            "translated_user": translated_message,
                            "language": selected_language,
                            "original_language": original_language,
                            "response": grouped_chunks
                        })
                
                except Exception as e:
                    st.error(f"Error processing request: {e}")
                    self.logger.error(f"Request processing error: {e}")
                    st.session_state.error_count += 1
            else:
                st.warning("Please enter a message before sending.")

def main():
    interface = MultilingualTranslatorInterface()
    interface.run()

if __name__ == "__main__":
    main()