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
import tabulate
from serpapi import GoogleSearch


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
    page_title="Enforcement - An Intelligent Enforcement Assistant",
    page_icon="imgs/avatar.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/david-sim",
        "Report a bug": "https://github.com/david-sim",
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

# Google Search results
def google_search_entity(address):
    print(f"Executing Google Search for: {address}")
    time.sleep(3)
    search = GoogleSearch({
        "q": f"{address}",
        "location": "Singapore",
        "hl": "en",
        "gl": "sg",
        "filter": "0",
        "api_key": SERPAPI_API_KEY
    })
    results = search.get_dict()

    if "error" in results:
        print(f"Error in SERP response: {results['error']}")
        return None

    organic_results = results.get("organic_results", [])

    return format_structured_results(parse_results(organic_results))

def parse_results(raw_results):
    return raw_results if isinstance(raw_results, list) else []

def format_structured_results(parsed_results):
    return "\n".join([
        f"Title: {item.get('title', '')}\nLink: {item.get('link', '')}\nSnippet: {item.get('snippet', '')}\nDate: {item.get('date', 'No date available')}\n"
        for item in parsed_results
    ])

@st.cache_data(show_spinner=False)
def on_chat_submit(chat_input):
    """
    Handle chat input submissions and interact with the OpenAI API.
    """
    user_input = chat_input.text.strip() if hasattr(chat_input, "text") and chat_input.text else ""
    user_input_lower = user_input.lower()

    # Track uploaded CSV in session state
    if "pending_csv" not in st.session_state:
        st.session_state.pending_csv = None
    if "pending_command" not in st.session_state:
        st.session_state.pending_command = None

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = initialize_conversation()

    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        assistant_reply = ""
        uploaded_file = chat_input.files[0] if hasattr(chat_input, "files") and chat_input.files else None

        # If only CSV is uploaded and no text command
        if uploaded_file is not None and not user_input:
            st.session_state.pending_csv = uploaded_file
            assistant_reply = (
                "You have uploaded a CSV file but did not specify whether it is for 'shophouse' or 'industrial'. "
                "Please type either 'shophouse' or 'industrial' in the chat input to proceed."
            )
        # If command is present ("shophouse" or "industrial") and CSV was previously uploaded
        elif ("shophouse" in user_input_lower or "industrial" in user_input_lower):
            # If CSV uploaded now
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    assistant_reply += f"\n\nDetected '{'shophouse' if 'shophouse' in user_input_lower else 'industrial'}' command with CSV uploaded."
                    assistant_reply += "\n\nCSV File Contents:\n"
                    assistant_reply += df.to_markdown(index=False)
                except Exception as e:
                    assistant_reply += f"\n\nError reading CSV file: {str(e)}"
            # If CSV was uploaded previously and command now provided
            elif st.session_state.pending_csv is not None:
                try:
                    st.session_state.pending_csv.seek(0)
                    df = pd.read_csv(st.session_state.pending_csv)
                    assistant_reply += f"\n\nDetected '{'shophouse' if 'shophouse' in user_input_lower else 'industrial'}' command with previously uploaded CSV."
                    assistant_reply += "\n\nCSV File Contents:\n"
                    assistant_reply += df.to_markdown(index=False)
                    st.session_state.pending_csv = None  # Clear after use
                except Exception as e:
                    assistant_reply += f"\n\nError reading CSV file: {str(e)}"
            else:
                tokens = user_input.split()
                if len(tokens) > 1:
                    address = " ".join(tokens[1:])
                    assistant_reply += f"\n\nDetected '{'shophouse' if 'shophouse' in user_input_lower else 'industrial'}' command with address: {address}"
                    assistant_reply += f"\n\nAddress Provided: {address}"
                else:
                    st.session_state.pending_command = user_input_lower
                    assistant_reply += "\n\nPlease provide an address after the command."
        # If user previously typed command and now provides address
        elif st.session_state.pending_command is not None and user_input:
            tokens = user_input.split()
            address = " ".join(tokens)
            assistant_reply += f"\n\nDetected '{st.session_state.pending_command}' command with address: {address}"
            assistant_reply += f"\n\nAddress Provided: {address}"
            st.session_state.pending_command = None  # Clear after use
        # Normal chat flow
        elif uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                assistant_reply += "\n\nCSV File Contents:\n"
                assistant_reply += df.to_markdown(index=False)
            except Exception as e:
                assistant_reply += f"\n\nError reading CSV file: {str(e)}"
        else:
            if not assistant_reply:
                assistant_reply = "Please enter a command or upload a file."

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
    mode = st.sidebar.radio("Select Mode:", options=["Google Search Results", "Enforcement"], index=1)

    st.sidebar.markdown("---")

    # Display basic interactions
    show_basic_info = st.sidebar.checkbox("Show Basic Interactions", value=True)
    if show_basic_info:
        st.sidebar.markdown("""
        ### Basic Interactions
        - **Ask About Addresses**: Upload a csv file containing Address and Approved Use. Type "shophouse" or "industrial" to scan for address. Alternatively, you may type in an address to search for an occupant.
        - **Google Snippet**: Look through Google Search results that were generated from the search.
        """)

    # Display advanced interactions
    show_advanced_info = st.sidebar.checkbox("Show Advanced Interactions", value=False)
    if show_advanced_info:
        st.sidebar.markdown("""
        ### Advanced Interactions
        - **CSV template**: Download this CSV template to populate your list of addresses and the corresponding approved use.
        - **Address Format**:  For entering address, please follow the standard address format: <block> <street> <unit> <postal code>
        - **Download CSV**: You can download CSV if there is a generated response of the address.
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