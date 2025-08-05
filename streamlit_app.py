import openai
import streamlit as st
import logging
import warnings
from PIL import Image
import time
import re
import base64
import pandas as pd
from openai import OpenAI, OpenAIError
import datetime
from langchain_openai import ChatOpenAI
from google_search import google_search_entity
from enforcement_processor import process_industrial_addresses_enhanced, process_csv

# Suppress langchain deprecation warnings
warnings.filterwarnings("ignore", message="Importing verbose from langchain root module is no longer supported.*")


# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
NUMBER_OF_MESSAGES_TO_DISPLAY = 20

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
    """Legacy callback function that immediately updates chat history."""
    st.session_state.history.append({"role": "assistant", "content": message})
    # Force a rerun to show the message immediately
    st.rerun()

def process_industrial_with_progress(addresses, llm):
    """
    Process industrial addresses with enhanced data collection and CSV generation.
    
    Args:
        addresses: List of addresses to process
        llm: Language model instance
    
    Returns:
        Tuple of (results, progress_messages, csv_buffer) - enhanced results with CSV download capability
    """
    try:
        # Use the enhanced processing approach with comprehensive data collection
        results, progress_messages, csv_buffer = process_industrial_addresses_enhanced(addresses, llm)
        
        return results, progress_messages, csv_buffer
    except Exception as e:
        logging.error(f"Error in enhanced processing: {str(e)}")
        # Return error result
        return [], [f"❌ Processing failed: {str(e)}"], None

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
        progress_messages_to_add = []  # Collect progress messages to add after main reply
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
                        # Call the enhanced processing function with CSV generation
                        results, progress_messages, csv_buffer = process_industrial_with_progress(addresses, llm)
                        
                        # Collect progress messages to add after main reply
                        progress_messages_to_add.extend(progress_messages)
                        
                        # Store CSV buffer for download instead of displaying results
                        if csv_buffer is not None:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"industrial_address_results_{timestamp}.csv"
                            st.session_state.csv_download_data = {
                                'buffer': csv_buffer,
                                'filename': filename,
                                'addresses_processed': len(addresses)
                            }
                            assistant_reply += f"\n\n✅ Successfully processed {len(addresses)} address(es). Results are ready for download as CSV file."
                        else:
                            assistant_reply += f"\n\n❌ Processing completed but CSV generation failed. Please check the logs."
                    else:
                        assistant_reply += "\n\nProcessed CSV Data:\n"
                        assistant_reply += f"{data}"
                except Exception as e:
                    assistant_reply += f"\n\nError reading CSV file: {str(e)}"
            # If CSV was uploaded previously and command now provided
            elif st.session_state.pending_csv is not None:
                try:
                    # Process CSV using csv_handler with the pending CSV file
                    data = process_csv(command, st.session_state.pending_csv)
                    assistant_reply += f"\n\nDetected '{command}' command with previously uploaded CSV."
                    if command == "industrial":
                        addresses = data["addresses"]
                        # Call the enhanced processing function with CSV generation
                        results, progress_messages, csv_buffer = process_industrial_with_progress(addresses, llm)
                        
                        # Collect progress messages to add after main reply
                        progress_messages_to_add.extend(progress_messages)
                        
                        # Store CSV buffer for download instead of displaying results
                        if csv_buffer is not None:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"industrial_address_results_{timestamp}.csv"
                            st.session_state.csv_download_data = {
                                'buffer': csv_buffer,
                                'filename': filename,
                                'addresses_processed': len(addresses)
                            }
                            assistant_reply += f"\n\n✅ Successfully processed {len(addresses)} address(es). Results are ready for download as CSV file."
                        else:
                            assistant_reply += f"\n\n❌ Processing completed but CSV generation failed. Please check the logs."
                    else:
                        assistant_reply += "\n\nProcessed CSV Data:\n"
                        assistant_reply += f"{data}"
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
                                # Add validation for required secrets
                                required_secrets = ['OPENAI_API_KEY', 'SERPAPI_API_KEY', 'LLM_MODEL']
                                missing_secrets = []
                                for secret in required_secrets:
                                    if not st.secrets.get(secret):
                                        missing_secrets.append(secret)
                                
                                if missing_secrets:
                                    assistant_reply += f"\n\n❌ Missing required configuration: {', '.join(missing_secrets)}"
                                else:
                                    # Call the enhanced processing logic
                                    results, progress_messages, csv_buffer = process_industrial_with_progress([address], llm)
                                    
                                    # Collect progress messages to add after main reply
                                    progress_messages_to_add.extend(progress_messages)
                                    
                                    # Store CSV buffer for download instead of displaying results
                                    if csv_buffer is not None:
                                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                        filename = f"industrial_address_result_{timestamp}.csv"
                                        st.session_state.csv_download_data = {
                                            'buffer': csv_buffer,
                                            'filename': filename,
                                            'addresses_processed': 1
                                        }
                                        assistant_reply += f"\n\n✅ Successfully processed address: {address}. Results are ready for download as CSV file."
                                    else:
                                        assistant_reply += f"\n\n❌ Processing completed but CSV generation failed. Please check the logs."
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
                        # Add validation for required secrets
                        required_secrets = ['OPENAI_API_KEY', 'SERPAPI_API_KEY', 'LLM_MODEL']
                        missing_secrets = []
                        for secret in required_secrets:
                            if not st.secrets.get(secret):
                                missing_secrets.append(secret)
                        
                        if missing_secrets:
                            assistant_reply += f"\n\n❌ Missing required configuration: {', '.join(missing_secrets)}"
                        else:
                            # Use the enhanced processing function
                            results, progress_messages, csv_buffer = process_industrial_with_progress([address], llm)
                            
                            # Collect progress messages to add after main reply
                            progress_messages_to_add.extend(progress_messages)
                            
                            # Store CSV buffer for download instead of displaying results
                            if csv_buffer is not None:
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"industrial_address_result_{timestamp}.csv"
                                st.session_state.csv_download_data = {
                                    'buffer': csv_buffer,
                                    'filename': filename,
                                    'addresses_processed': 1
                                }
                                assistant_reply += f"\n\n✅ Successfully processed address: {address}. Results are ready for download as CSV file."
                            else:
                                assistant_reply += f"\n\n❌ Processing completed but CSV generation failed. Please check the logs."
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
        
        # Add collected progress messages after the main reply
        for message in progress_messages_to_add:
            st.session_state.history.append({"role": "assistant", "content": message})

    except OpenAIError as e:
        logging.error(f"Error occurred: {e}")
        st.error(f"OpenAI Error: {str(e)}")

def initialize_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'cleanup_progress' not in st.session_state:
        st.session_state.cleanup_progress = False
    if 'csv_download_data' not in st.session_state:
        st.session_state.csv_download_data = None

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

        # CSV Download Section
        if hasattr(st.session_state, 'csv_download_data') and st.session_state.csv_download_data is not None:
            csv_data = st.session_state.csv_download_data
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### 📊 Processing Results Ready")
                st.markdown(f"**Addresses Processed:** {csv_data['addresses_processed']}")
                st.markdown(f"**Filename:** {csv_data['filename']}")
                
                # Reset buffer position for download
                csv_data['buffer'].seek(0)
                
                if st.download_button(
                    label="📥 Download CSV Results",
                    data=csv_data['buffer'].getvalue(),
                    file_name=csv_data['filename'],
                    mime="text/csv",
                    key="download_csv_results"
                ):
                    st.success("✅ CSV file downloaded successfully!")
                    # Clear the download data after successful download
                    st.session_state.csv_download_data = None
                    st.rerun()
            st.markdown("---")

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