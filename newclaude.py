import streamlit as st
import speech_recognition as sr
from interpreter import interpreter
import logging
from typing import Dict, List

class UnifiedNLInterface:
    def __init__(self):
        """
        Initialize the Unified Natural Language Interface application.
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

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
            "error_count"
        ]
        for var in session_vars:
            if var not in st.session_state:
                st.session_state[var] = "" if var != "chat_history" else []
                if var == "error_count":
                    st.session_state[var] = 0

    def get_voice_input(self) -> str:
        """
        Capture voice input using speech recognition.
        
        Returns:
            str: Transcribed voice message
        """
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        
        with sr.Microphone() as source:
            st.info("Listening... Speak into the microphone")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                st.success("Voice input captured. Processing...")
                
                # Use Google Speech Recognition with error handling
                voice_text = recognizer.recognize_google(audio)
                self.logger.info("Voice input successfully transcribed")
                return voice_text
            
            except sr.WaitTimeoutError:
                st.warning("Listening timed out. Please try again.")
            except sr.UnknownValueError:
                st.error("Could not understand the audio. Please speak clearly.")
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
        Main application runner with improved UI and error handling.
        """
        st.set_page_config(
            page_title="Unified NL Interface", 
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🚀 Unified Natural Language Interface")
        
        # Sidebar for configuration and metadata
        with st.sidebar:
            st.header("🔧 System Configuration")
            st.json({
                "Model": interpreter.llm.model,
                "API Base": interpreter.llm.api_base,
                "Offline Mode": interpreter.offline
            })
            
            st.divider()
            st.metric("Total Errors", st.session_state.error_count)

        # Main interaction area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            message = st.text_input(
                "Enter your message:", 
                value=st.session_state.voice_message, 
                placeholder="Type your command or query..."
            )
        
        with col2:
            st.write("") # Vertical alignment hack
            voice_button = st.button("🎙️ Voice Input")

        if voice_button:
            voice_message = self.get_voice_input()
            if voice_message:
                st.session_state.voice_message = voice_message
                st.rerun()

        # Message sending logic
        if st.button("Send Message", type="primary"):
            if message.strip():
                try:
                    with st.spinner("Processing your request..."):
                        response_stream = interpreter.chat(message, stream=True)
                        
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
                            "response": grouped_chunks
                        })
                
                except Exception as e:
                    st.error(f"Error processing request: {e}")
                    self.logger.error(f"Request processing error: {e}")
                    st.session_state.error_count += 1
            else:
                st.warning("Please enter a message before sending.")

def main():
    interface = UnifiedNLInterface()
    interface.run()

if __name__ == "__main__":
    main()