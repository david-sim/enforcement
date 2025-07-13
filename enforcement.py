import openai
import streamlit as st
import logging
from PIL import Image, ImageEnhance
import time
import json
import requests
import base64
import pandas as pd
from openai import OpenAI, OpenAIError

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
NUMBER_OF_MESSAGES_TO_DISPLAY = 20
API_DOCS_URL = "https://docs.streamlit.io/library/api-reference"

# Retrieve and validate API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    st.error("Please add your OpenAI API key to the Streamlit secrets.toml file.")
    st.stop()

# Assign OpenAI API Key
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Streamly - An Intelligent Streamlit Assistant",
    page_icon="imgs/avatar.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/",
        "Report a bug": "https://github.com/",
        "About": """
            ## Enforcement Assistant
            ### Powered using GPT-4o-mini

            The AI Assistant aims to provide address resolution,
            and answer questions about addresses format, and more.
        """
    }
)

# Streamlit Title
st.title("Enforcement Assistant")

def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def long_running_task(duration):
    """
    Simulates a long-running operation.

    Parameters:
    - duration: int, duration of the task in seconds

    Returns:
    - str: Completion message
    """
    time.sleep(duration)
    return "Long-running operation completed."

@st.cache_data(show_spinner=True)
def load_and_enhance_image(image_path, enhance=False):
    """
    Load and optionally enhance an image.

    Parameters:
    - image_path: str, path of the image
    - enhance: bool, whether to enhance the image or not

    Returns:
    - img: PIL.Image.Image, (enhanced) image
    """
    img = Image.open(image_path)
    if enhance:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
    return img



def display_streamlit_updates():
    """Display the latest updates of the Streamlit."""
    with st.expander("Streamlit 1.36 Announcement", expanded=False):
        st.markdown("For more details on this version, check out the [Streamlit Forum post](https://docs.streamlit.io/library/changelog#version).")

def initialize_conversation():
    """
    Initialize the conversation history with system and assistant messages.

    Returns:
    - list: Initialized conversation history.
    """
    assistant_message = "Hello! How can I assist you today?"

    conversation_history = [
        {"role": "system", "content": "You are trained in Singapore addresses."},
        {"role": "system", "content": "You are familiar with both Singapore's industrial and shophouse addresses."},
        {"role": "system", "content": "Refer to conversation history to provide context to your response."},
        {"role": "assistant", "content": assistant_message}
    ]
    return conversation_history

@st.cache_data(show_spinner=False)
def on_chat_submit(chat_input):
    """
    Handle chat input submissions and interact with the OpenAI API.
    """
    user_input = chat_input.text.strip().lower() if hasattr(chat_input, "text") and chat_input.text else ""

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = initialize_conversation()

    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        model_engine = "gpt-4o-mini"
        assistant_reply = ""
        #st.write("chat_input:", chat_input)
        #st.write("chat_input.files:", getattr(chat_input, "files", None))

        # Handle CSV file input if present
        uploaded_file = chat_input.files[0] if hasattr(chat_input, "files") and chat_input.files else None
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)  # Ensure pointer is at start
                df = pd.read_csv(uploaded_file)
                preview = df.head().to_markdown(index=False)
                assistant_reply += f"\n\nCSV Preview:\n{preview}"
            except Exception as e:
                assistant_reply += f"\n\nError reading CSV file: {str(e)}"

        if "latest updates" in user_input:
            assistant_reply = "Here are the latest highlights.\n" + assistant_reply
        elif user_input:
            response = client.chat.completions.create(
                model=model_engine,
                messages=st.session_state.conversation_history
            )
            assistant_reply = response.choices[0].message.content + assistant_reply
        else:
            if not assistant_reply:
                assistant_reply = "Please enter a question or upload a file."

        st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    except OpenAIError as e:
        logging.error(f"Error occurred: {e}")
        st.error(f"OpenAI Error: {str(e)}")


def initialize_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def main():
    """
    Display Streamlit updates and handle the chat interface.
    """
    initialize_session_state()

    if not st.session_state.history:
        initial_bot_message = "Hello! How can I assist you today?"
        st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
        #st.session_state.conversation_history = initialize_conversation()

    # Load and display sidebar image
    img_path = "imgs/avatar.png"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}">',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    # Sidebar for Mode Selection
    mode = st.sidebar.radio("Select Mode:", options=["Google Map", "Enforcement"], index=1)

    st.sidebar.markdown("---")

    # Display basic interactions
    show_basic_info = st.sidebar.checkbox("Show Basic Interactions", value=True)
    if show_basic_info:
        st.sidebar.markdown("""
        ### Basic Interactions
        - **Ask About Streamlit**: Type your questions about Streamlit's latest updates, features, or issues.
        - **Search for Code**: Use keywords like 'code example', 'syntax', or 'how-to' to get relevant code snippets.
        - **Navigate Updates**: Switch to 'Updates' mode to browse the latest Streamlit updates in detail.
        """)

    # Display advanced interactions
    show_advanced_info = st.sidebar.checkbox("Show Advanced Interactions", value=False)
    if show_advanced_info:
        st.sidebar.markdown("""
        ### Advanced Interactions
        - **Generate an App**: Use keywords like **generate app**, **create app** to get a basic Streamlit app code.
        - **Code Explanation**: Ask for **code explanation**, **walk me through the code** to understand the underlying logic of Streamlit code snippets.
        - **Project Analysis**: Use **analyze my project**, **technical feedback** to get insights and recommendations on your current Streamlit project.
        - **Debug Assistance**: Use **debug this**, **fix this error** to get help with troubleshooting issues in your Streamlit app.
        """)

    st.sidebar.markdown("---")

    # Load and display image with glowing effect
    img_path = "imgs/stsidebarimg.png"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}">',
            unsafe_allow_html=True,
        )
    
    if mode == "Enforcement":
        chat_input = st.chat_input("Ask me about addresses:", accept_file=True, file_type="csv")
        #st.write("chat_input type:", type(chat_input))
        #st.write("chat_input dir:", dir(chat_input))
        #st.write("chat_input value:", chat_input)
        #st.write("chat_input attributes:", dir(chat_input))

        if chat_input is not None:
            user_message = chat_input.text  # The user's text input
            on_chat_submit(chat_input)

        # Display chat history
        for message in st.session_state.history[-NUMBER_OF_MESSAGES_TO_DISPLAY:]:
            role = message["role"]
            avatar_image = "imgs/avatar.png" if role == "assistant" else "imgs/stuser.png" if role == "user" else None
            with st.chat_message(role, avatar=avatar_image):
                st.write(message["content"])

    else:
        display_streamlit_updates()

if __name__ == "__main__":
    main()