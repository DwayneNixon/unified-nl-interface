import streamlit as st
from interpreter import interpreter
import speech_recognition as sr

# Set up the Open-Interpreter instance
interpreter.offline = True  # Disable online features
interpreter.llm.model = "ollama/llama3.1"  # Use the specified model
interpreter.llm.api_base = "http://localhost:11434"  # Set the API endpoint

st.title("Unified NL Interface")

# Function to display grouped content
def display_grouped_content(grouped_chunks):
    """
    Displays grouped chunks based on their type in Streamlit.
    """
    for chunk_type, content_list in grouped_chunks.items():
        if not content_list:  # Skip if no content
            continue

        # Handle 'code' type specifically to avoid splitting lines unnecessarily
        if chunk_type == "code":
            combined_code = "".join(content_list)  # Combine without adding newlines
            st.code(combined_code.strip(), language="python")
        else:
            # Combine all other types with line breaks
            combined_content = "\n".join(content_list).strip()

            # Display based on type
            if chunk_type == "console":
                st.text_area("Console Output:", combined_content, height=150, disabled=True)
            elif chunk_type == "confirmation":
                st.success(combined_content)
            elif chunk_type == "message":
                st.write(combined_content)
            else:
                st.warning(f"Unknown type: {chunk_type}\n{combined_content}")

# Function to capture voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak into the microphone.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.success("Voice input captured. Processing...")
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            st.error("Listening timed out. Please try again.")
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"Error with the speech recognition service: {e}")
    return ""

# User input for the message
message = st.text_input("Enter your message:", placeholder="Type your message here...")

# Voice input button
if st.button("Use Voice Input"):
    voice_message = get_voice_input()
    if voice_message:
        message = voice_message
        st.write(f"Voice input recognized: {message}")

# Main UI logic
if st.button("Send Message"):
    if message.strip():  # Ensure message is not empty
        st.write(f"Sending message: {message}")
        response_stream = interpreter.chat(message, stream=True)  # Get the streamed response

        # Initialize a dictionary to group chunks by type
        grouped_chunks = {
            "code": [],
            "console": [],
            "confirmation": [],
            "message": [],
        }

        # Process and group the streamed chunks
        try:
            for chunk in response_stream:
                chunk_type = chunk.get("type", "message")  # Default to 'message'
                content = chunk.get("content", "")  # Default to empty string

                # Skip blank or invalid content
                if isinstance(content, str) and content.strip():
                    grouped_chunks.setdefault(chunk_type, []).append(content)

            # Display the grouped content
            display_grouped_content(grouped_chunks)

        except Exception as e:
            st.error(f"Error processing response: {e}")
    else:
        st.warning("Please enter a message before sending.")
