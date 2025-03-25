import streamlit as st
import speech_recognition as sr
from interpreter import interpreter
import traceback

class OpenInterpreterApp:
    def __init__(self):
        """
        Initialize the Streamlit app with configuration and setup.
        """
        # Configure Open Interpreter
        interpreter.offline = True
        interpreter.llm.model = "ollama/llama3.1"
        interpreter.llm.api_base = "http://localhost:11434"
        
        # Initialize session state
        self._initialize_session_state()
        
        # Set up page configuration
        st.set_page_config(
            page_title="Unified Natural Language Interface", 
            page_icon=":robot_face:", 
            layout="wide"
        )

    def _initialize_session_state(self):
        """
        Initialize or reset session state variables.
        """
        default_states = {
            "voice_message": "",
            "conversation_history": [],
            "error_message": None
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def get_voice_input(self):
        """
        Capture voice input using speech recognition.
        
        Returns:
            str: Transcribed voice message
        """
        recognizer = sr.Recognizer()
        
        # Adjust for ambient noise before listening
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            with sr.Microphone() as source:
                st.info("🎤 Listening... Speak clearly into the microphone.")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
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

    def display_grouped_content(self, grouped_chunks):
        """
        Display content from different response chunks.
        
        Args:
            grouped_chunks (dict): Grouped response content
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

    def run(self):
        """
        Main application runner with UI components.
        """
        st.title("🤖 Unified Natural Language Interface")
        
        # Sidebar for configuration and instructions
        with st.sidebar:
            st.header("📝 Instructions")
            st.markdown("""
            - Type or use voice input to interact
            - Support for code generation, console commands
            - Powered by Open Interpreter
            """)
            
            # Model selection (if multiple models are available)
            model_options = ["ollama/llama3.1", "other_model"]
            selected_model = st.selectbox("Select LLM Model", model_options)
            if selected_model != interpreter.llm.model:
                interpreter.llm.model = selected_model

        # Input section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            message = st.text_input(
                "Enter your message:", 
                value=st.session_state.voice_message, 
                key="message_input",
                placeholder="Type a command or describe a task..."
            )
        
        with col2:
            st.write("") # Spacer
            voice_input_button = st.button("🎤 Voice Input")
        
        # Voice input handling
        if voice_input_button:
            voice_message = self.get_voice_input()
            if voice_message:
                st.session_state.voice_message = voice_message
                st.experimental_rerun()
        
        # Send message button
        if st.button("🚀 Send Message", use_container_width=True):
            if message.strip():
                try:
                    # Reset previous error
                    st.session_state.error_message = None
                    
                    # Stream response from Open Interpreter
                    grouped_chunks = {
                        "code": [],
                        "console": [],
                        "confirmation": [],
                        "message": [],
                    }
                    
                    with st.spinner("Processing your request..."):
                        response_stream = interpreter.chat(message, stream=True)
                        
                        for chunk in response_stream:
                            chunk_type = chunk.get("type", "message")
                            content = chunk.get("content", "")
                            
                            if isinstance(content, str) and content.strip():
                                grouped_chunks.setdefault(chunk_type, []).append(content)
                        
                        # Display results
                        self.display_grouped_content(grouped_chunks)
                    
                    # Clear voice message after successful send
                    st.session_state.voice_message = ""
                
                except Exception as e:
                    error_details = traceback.format_exc()
                    st.session_state.error_message = f"Error: {e}"
                    st.error(f"An error occurred: {e}")
                    st.expander("Error Details").code(error_details)
            else:
                st.warning("Please enter a message before sending.")

        # Display conversation history or error (optional enhancement)
        if st.session_state.error_message:
            st.error(st.session_state.error_message)

def main():
    app = OpenInterpreterApp()
    app.run()

if __name__ == "__main__":
    main()