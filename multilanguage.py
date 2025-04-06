import streamlit as st
import speech_recognition as sr
from interpreter import interpreter
import logging
from typing import Dict, List

class MultilingualNLInterface:
    def __init__(self):
        """
        Initialize the Multilingual Natural Language Interface application.
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Configure language mappings
        self.language_codes = {
            "English": "en-US",
            "Hindi": "hi-IN",
            # "Marathi": "mr-IN",
            # "Bengali": "bn-IN", 
            # "Tamil": "ta-IN",
            # "Telugu": "te-IN",
            # "Kannada": "kn-IN",
            # "Malayalam": "ml-IN",
            # "Gujarati": "gu-IN",
            # "Punjabi": "pa-IN"
        }

        # Configure Open Interpreter
        self._configure_interpreter()
        
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
            "chat_history", 
            "error_count",
            "selected_language"
        ]
        for var in session_vars:
            if var not in st.session_state:
                if var == "voice_message":
                    st.session_state[var] = ""
                elif var == "chat_history":
                    st.session_state[var] = []
                elif var == "error_count":
                    st.session_state[var] = 0
                elif var == "selected_language":
                    st.session_state[var] = "English"

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
        """
        content_type_map = {
            "code": (st.code, {"language": "python"}),
            "console": (lambda content: st.text_area("Console Output:", content, height=200, disabled=True), {}),
            "confirmation": (st.success, {}),
            "message": (st.markdown, {})
        }

        for chunk_type, content_list in grouped_chunks.items():
            if not content_list:
                continue  # Skip empty sections

            renderer, render_kwargs = content_type_map.get(
                chunk_type, 
                (st.warning, {"icon": "⚠️"})
            )

            # Join content properly
            combined_content = "\n\n".join(content_list).strip()

            # 🔹 Ensure Markdown renders correctly
            if chunk_type == "message":
                st.markdown(combined_content, unsafe_allow_html=False)
            else:
                renderer(combined_content, **render_kwargs)

            # 🔹 Add a visual break for clarity
            st.markdown("---")


    def run(self):
        """
        Main application runner with multilingual support.
        """
        st.set_page_config(
            page_title="Unified NL Interface", 
            page_icon="🌐",
            layout="wide"
        )
        
        st.title("🚀 Unified Natural Language Interface")
        
        # Sidebar for configuration and metadata
        # with st.sidebar:
        #     st.header("🔧 System Configuration")
        #     st.json({
        #         "Model": interpreter.llm.model,
        #         "API Base": interpreter.llm.api_base,
        #         "Offline Mode": interpreter.offline
        #     })
            
        #     st.divider()
        #     st.metric("Total Errors", st.session_state.error_count)

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
                        # Attempt to process in target language
                        response_stream = interpreter.chat(message, stream=True)
                        
                        grouped_chunks = {
                            "code": [],
                            "console": [],
                            "confirmation": [],
                            "message": []
                        }

                        # Buffer for incomplete messages
                        partial_content = {key: "" for key in grouped_chunks}

                        for chunk in response_stream:
                            chunk_type = chunk.get("type", "message")
                            content = chunk.get("content", "")

                            if isinstance(content, str) and content.strip():
                                if chunk_type in grouped_chunks:
                                    # Accumulate content until a full line is formed
                                    partial_content[chunk_type] += content

                                    # If we detect a full sentence (e.g., newline or period), store it
                                    if content.endswith("\n") or content.endswith("."):
                                        grouped_chunks[chunk_type].append(partial_content[chunk_type].strip())
                                        partial_content[chunk_type] = ""  # Reset buffer
                                else:
                                    print(f"Unexpected chunk type: {chunk_type}")  # Handle unknown types

                        # Store any remaining buffered content
                        for chunk_type, text in partial_content.items():
                            if text.strip():
                                grouped_chunks[chunk_type].append(text.strip())

                        self.display_grouped_content(grouped_chunks)
                        
                        # Optional: Update chat history
                        st.session_state.chat_history.append({
                            "user": message,
                            "language": selected_language,
                            "response": grouped_chunks
                        })
                
                except Exception as e:
                    st.error(f"Error processing request: {e}")
                    self.logger.error(f"Request processing error: {e}")
                    st.session_state.error_count += 1
            else:
                st.warning("Please enter a message before sending.")

def main():
    interface = MultilingualNLInterface()
    interface.run()

if __name__ == "__main__":
    main()