import openai
import streamlit as st
import logging
import warnings
from PIL import Image, ImageEnhance
import json
import time
import re
import requests
import base64
import pandas as pd
from openai import OpenAI, OpenAIError
from langchain_openai import ChatOpenAI
#from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import tabulate
from google_search import google_search_entity, parse_results, format_structured_results
from csv_handler import process_csv, process_industrial_addresses

# Suppress langchain deprecation warnings
warnings.filterwarnings("ignore", message="Importing verbose from langchain root module is no longer supported.*")


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

llm = ChatOpenAI(model=st.secrets.get("LLM_MODEL",None), temperature=st.secrets.get("LLM_TEMPERATURE",None))

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
            ### Powered using GPT

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
        #{"role": "system", "content": "You are trained in Singapore addresses."},
        {"role": "assistant", "content": assistant_message}
    ]
    return conversation_history

def is_valid_address(address):
    """
    Validates Singapore address format: <block> <street> <unit> <postal code>
    Example: 137 Amoy Street #01-05 069965
    """
    # Simple regex for block, street, unit, and 6-digit postal code
    pattern = r"^\d+\s+[\w\s]+(?:#\d{2}-\d{2})?\s+\d{6}$"
    return re.match(pattern, address.strip()) is not None

def chat_callback(message):
    #st.chat_message("assistant").write(message)
    st.session_state.history.append({"role": "assistant", "content": message})

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

    #if 'conversation_history' not in st.session_state:
    #    st.session_state.conversation_history = initialize_conversation()

    #st.session_state.conversation_history.append({"role": "user", "content": user_input})

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
            command = "shophouse" if "shophouse" in user_input_lower else "industrial"
            if uploaded_file is not None:
                try:
                    data = process_csv(command, uploaded_file)
                    assistant_reply += f"\n\nDetected '{command}' command with CSV uploaded."
                    if command == "industrial":
                        addresses = data["addresses"]
                        # Call the new industrial processing function
                        results = process_industrial_addresses(addresses, llm, chat_callback=chat_callback)
                        # Format results for display
                        assistant_reply += "\n\nIndustrial Address Results:\n"
                        for row in results:
                            assistant_reply += f"\nAddress: {row[0]}\n\nOccupant: {row[1]}\n\nAnalysis: {row[2]}\n\nSearch Results: \n\n{row[3]}\n---"
                    else:
                        assistant_reply += "\n\nProcessed CSV Data:\n"
                        assistant_reply += f"{data}"
                except Exception as e:
                    assistant_reply += f"\n\nError reading CSV file: {str(e)}"
            # If CSV was uploaded previously and command now provided
            elif st.session_state.pending_csv is not None:
                try:
                    # Process CSV using csv_handler
                    data = process_csv(command, uploaded_file)
                    assistant_reply += f"\n\nDetected '{command}' command with CSV uploaded."
                    assistant_reply += "\n\nProcessed CSV Data:\n"
                    assistant_reply += f"{data}"
                    #st.session_state.pending_csv.seek(0)
                    #df = pd.read_csv(st.session_state.pending_csv)
                    #assistant_reply += f"\n\nDetected '{'shophouse' if 'shophouse' in user_input_lower else 'industrial'}' command with previously uploaded CSV."
                    #assistant_reply += "\n\nCSV File Contents:\n"
                    #assistant_reply += df.to_markdown(index=False)
                    st.session_state.pending_csv = None  # Clear after use
                except Exception as e:
                    assistant_reply += f"\n\nError reading CSV file: {str(e)}"
            else:
                tokens = user_input.split()
                if len(tokens) > 1:
                    address = " ".join(tokens[1:])
                    command = "shophouse" if "shophouse" in user_input_lower else "industrial"
                    assistant_reply += f"\n\nDetected '{command}' command with address: {address}"
                    assistant_reply += f"\n\nAddress Provided: {address}"
                    
                    # Process the address immediately (same logic as the pending command case)
                    if is_valid_address(address):
                        if command == "industrial":
                            try:
                                assistant_reply += "\n\n🔄 Processing industrial address..."
                                
                                # Add validation for required secrets
                                required_secrets = ['OPENAI_API_KEY', 'SERPAPI_API_KEY', 'LLM_MODEL']
                                missing_secrets = []
                                for secret in required_secrets:
                                    if not st.secrets.get(secret):
                                        missing_secrets.append(secret)
                                
                                if missing_secrets:
                                    assistant_reply += f"\n\n❌ Missing required configuration: {', '.join(missing_secrets)}"
                                else:
                                    assistant_reply += "\n\n✅ Configuration validated. Calling processing function..."
                                    
                                    # Call the processing logic
                                    results = process_industrial_addresses([address], llm, chat_callback=chat_callback)
                                    assistant_reply += f"\n\n✅ Processing completed. Found {len(results)} result(s)."
                                    for row in results:
                                        assistant_reply += f"\n\nAddress: {row[0]}\n\nOccupant: {row[1]}\n\nAnalysis: {row[2]}\n\nSearch Results: \n\n{row[3]}\n---"
                            except Exception as e:
                                assistant_reply += f"\n\n❌ Error processing industrial address: {str(e)}"
                                logging.error(f"Error in process_industrial_addresses: {str(e)}")
                                import traceback
                                logging.error(f"Full traceback: {traceback.format_exc()}")
                        # Add similar logic for "shophouse" if needed
                    else:
                        assistant_reply += f"\n\n❌ The address format is invalid. Please use: <block> <street> <unit> <postal code> (e.g., 137 Amoy Street #01-05 069965)"
                else:
                    st.session_state.pending_command = user_input_lower
                    assistant_reply += "\n\nPlease provide an address after the command."
        # If user previously typed command and now provides address
        elif st.session_state.pending_command is not None and user_input:
            tokens = user_input.split()
            address = " ".join(tokens)
            command = st.session_state.pending_command
            if is_valid_address(address):
                assistant_reply += f"\n\nDetected '{command}' command with address: {address}"
                assistant_reply += f"\n\nAddress Provided: {address}"
                # Process the address as needed (e.g., searchprocess_industrial_addresses or further logic)
                if command == "industrial":
                    try:
                        assistant_reply += "\n\n🔄 Processing industrial address..."
                        
                        # Add validation for required secrets
                        required_secrets = ['OPENAI_API_KEY', 'SERPAPI_API_KEY', 'LLM_MODEL']
                        missing_secrets = []
                        for secret in required_secrets:
                            if not st.secrets.get(secret):
                                missing_secrets.append(secret)
                        
                        if missing_secrets:
                            assistant_reply += f"\n\n❌ Missing required configuration: {', '.join(missing_secrets)}"
                        else:
                            assistant_reply += "\n\n✅ Configuration validated. Calling processing function..."
                            
                            # You can call your processing logic here, e.g.:
                            results = process_industrial_addresses([address], llm, chat_callback=chat_callback)
                            assistant_reply += f"\n\n✅ Processing completed. Found {len(results)} result(s)."
                            for row in results:
                                assistant_reply += f"\n\nAddress: {row[0]}\n\nOccupant: {row[1]}\n\nAnalysis: {row[2]}\n\nSearch Results: \n\n{row[3]}\n---"
                    except Exception as e:
                        assistant_reply += f"\n\n❌ Error processing industrial address: {str(e)}"
                        logging.error(f"Error in process_industrial_addresses: {str(e)}")
                        import traceback
                        logging.error(f"Full traceback: {traceback.format_exc()}")
                    # Add similar logic for "shophouse" if needed
            else:
                assistant_reply += f"\n\n❌ The address format is invalid. Please use: <block> <street> <unit> <postal code> (e.g., 137 Amoy Street #01-05 069965)"
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

        #st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
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
        st.session_state.history = initialize_conversation()

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