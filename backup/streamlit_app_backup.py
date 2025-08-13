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

# Initialize LLM with config from config.json
try:
    from config_manager import get_llm_model, get_llm_temperature
    llm = ChatOpenAI(
        model=get_llm_model(),
        temperature=float(get_llm_temperature()),
        api_key=st.secrets.get("OPENAI_API_KEY")
    )
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
    # Fallback to default values
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.78,
        api_key=st.secrets.get("OPENAI_API_KEY")
    )

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

def process_csv_with_ui(uploaded_file, address_type, llm):
    """
    Process CSV file with real-time UI progress display and return downloadable results.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        address_type: String indicating "shophouse" or "industrial"  
        llm: Language model instance
        
    Returns:
        Tuple of (results, csv_buffer) or None if processing fails
    """
    try:
        # Import processing functions
        from enforcement_processor import process_csv
        
        # Process the CSV to extract addresses
        csv_data = process_csv(address_type, uploaded_file)
        addresses = csv_data.get("addresses", [])
        
        if not addresses:
            st.error("No addresses found in the uploaded file.")
            return None
            
        # Display initial info
        st.info(f"Found {len(addresses)} addresses to process")
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        progress_log = st.empty()
        
        # Process addresses based on type
        if address_type.lower() == "industrial":
            # Use modified industrial processing with real-time callbacks
            results, csv_buffer = process_industrial_addresses_with_realtime_ui(
                addresses, llm, progress_bar, status_text, progress_log
            )
            
            # Final success message
            progress_bar.progress(1.0)
            status_text.success(f"✅ Processing completed! Processed {len(results)} addresses.")
            
            return results, csv_buffer
        else:
            st.warning(f"Processing for {address_type} is not yet implemented.")
            return None
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def process_industrial_addresses_with_realtime_ui(addresses, llm, progress_bar, status_text, progress_log):
    """
    Process industrial addresses with real-time UI updates.
    
    Args:
        addresses: List of addresses to process
        llm: Language model instance
        progress_bar: Streamlit progress bar component
        status_text: Streamlit text component for status
        progress_log: Streamlit component for progress log
        
    Returns:
        Tuple of (results, csv_buffer)
    """
    import datetime
    import re
    import io
    import pandas as pd
    
    results = []
    progress_messages = []
    total_addresses = len(addresses)
    
    def log_progress_realtime(message, current_index=None):
        """Log progress with real-time UI updates."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Always log to console
        print(formatted_message)
        
        # Add to progress messages
        progress_messages.append(formatted_message)
        
        # Update progress bar if index provided
        if current_index is not None:
            progress = (current_index) / total_addresses
            progress_bar.progress(progress)
            status_text.info(f"Processing {current_index}/{total_addresses}: {message}")
        
        # Update progress log (show all messages with scrollbar)
        progress_log.text_area(
            "Processing Log",
            value="\n".join(progress_messages),
            height=200,
            disabled=True,
            key=f"progress_log_{len(progress_messages)}"
        )
    
    # Get system rules
    try:
        from enforcement_processor import get_occupant_rules, get_compliance_rules, create_csv_for_download
        occupant_rules = get_occupant_rules()
        compliance_rules = get_compliance_rules()
    except Exception as e:
        st.error(f"Failed to load processing rules: {str(e)}")
        return [], None
    
    today = datetime.date.today().isoformat()
    
    log_progress_realtime(f"🚀 Starting enhanced processing of {total_addresses} address(es)...")
    
    for i in range(len(addresses)):
        try:
            address = addresses[i]
            current_step = i + 1
            log_progress_realtime(f"📍 Processing {address}", current_step)
            
            # Initialize variables
            address_search_results_raw = ""
            address_search_results_raw_variant = ""
            confirmed_occupant_google_search_results = ""
            verified_occupant_response = ""
            verification_analysis = ""
            confirmed_occupant = ""
            compliance_level = "Need more information"
            rationale = "Unable to confirm occupant, compliance assessment not performed."
            
            # Step 1: Google search for address
            log_progress_realtime(f"🔍 Searching for address information...", current_step)
            
            try:
                from google_search import google_search_entity
                
                # Original address search
                address_search_query = f"{address}"
                address_search_results_raw = google_search_entity(address_search_query)
                
                # Variant address search
                address_search_query_variant = f"address {address}"
                address_search_results_raw_variant = google_search_entity(address_search_query_variant)
                
                # Check if original search failed
                if address_search_results_raw is None:
                    address_search_results_raw = "No address search results available"
                
                # Check if variant search failed
                if address_search_results_raw_variant is None:
                    address_search_results_raw_variant = "No variant address search results available"
                
                # Log success if at least one search succeeded
                if address_search_results_raw != "No address search results available" or address_search_results_raw_variant != "No variant address search results available":
                    log_progress_realtime(f"✅ Address search completed", current_step)
                else:
                    log_progress_realtime(f"❌ Both address searches failed", current_step)
                        
            except Exception as search_error:
                address_search_results_raw = f"Address search failed: {str(search_error)}"
                address_search_results_raw_variant = f"Variant address search failed: {str(search_error)}"
                log_progress_realtime(f"❌ Address search failed", current_step)
            
            # Step 2: LLM Analysis for occupant identification
            log_progress_realtime(f"🤖 Analyzing occupant...", current_step)
            
            from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
            
            occupant_prompt = f"""
Today's date is {today}
Identify the current occupant of: {address} using the search results below.

<google_search_results_original>
{address_search_results_raw}
</google_search_results_original>

<google_search_results_variant>
{address_search_results_raw_variant}
</google_search_results_variant>

---

### FORMAT

Selected Occupant: <Business name from snippets or "Need more information">
---End of Confirmed Occupant---

Verification Analysis:
- Matched snippet(s): (Quote the relevant snippets that matches address, -url, -label credible/non-credible source and -source date)
- Business Summary: (Summarise the core and other business activities of the selected occupant)
- Reasoning: (Show your responses for each step. Why that entity was chosen, or why no match could be confirmed)
---End of Verification---
"""

            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(occupant_rules),
                HumanMessagePromptTemplate.from_template(occupant_prompt)
            ])

            # Runnable chain
            occupant_chain = prompt | llm

            try:
                # Step 1: Identify the occupant and verification analysis
                verified_occupant_response = occupant_chain.invoke({}).content.strip()
                log_progress_realtime(f"✅ Occupant analysis completed", current_step)

                # Parse the response using regex
                confirmed_occupant_match = re.search(r"Selected Occupant:\s*(.*?)\s*---End of Confirmed Occupant---", verified_occupant_response, re.DOTALL)
                confirmed_occupant = confirmed_occupant_match.group(1).strip() if confirmed_occupant_match else "Need more information"

                verification_analysis_match = re.search(r"Verification Analysis:\s*(.*?)\s*---End of Verification---", verified_occupant_response, re.DOTALL)
                verification_analysis = verification_analysis_match.group(1).strip() if verification_analysis_match else "Analysis not available"
                
            except Exception as llm_error:
                log_progress_realtime(f"❌ Analysis failed", current_step)
                confirmed_occupant = "Error"
                verification_analysis = f"LLM analysis failed: {str(llm_error)}"

            # Step 3: Compliance assessment if occupant is identified
            if confirmed_occupant == "Need more information":
                compliance_level = "Need more information"
                rationale = "Unable to confirm occupant, compliance assessment not performed."
                confirmed_occupant_google_search_results = "No occupant identified for search"
            else:
                log_progress_realtime(f"🔍 Searching for occupant information...", current_step)

                # Step 2: Google Search for Occupant
                try:
                    from google_search import google_search_entity
                    confirmed_occupant_google_search_results = google_search_entity(confirmed_occupant)
                    
                    if confirmed_occupant_google_search_results is None:
                        confirmed_occupant_google_search_results = "No occupant search results available"
                        log_progress_realtime(f"⚠️ Occupant search failed", current_step)
                    else:
                        log_progress_realtime(f"✅ Occupant search completed", current_step)
                            
                except Exception as search_error:
                    confirmed_occupant_google_search_results = f"Occupant search failed: {str(search_error)}"
                    log_progress_realtime(f"⚠️ Occupant search failed", current_step)

                # Step 3: Compliance assessment
                log_progress_realtime(f"⚖️ Assessing compliance...", current_step)
                
                compliance_prompt = f"""
Assess the occupant's operations based on the following information:

### Selected Occupant: {confirmed_occupant}

Google Search Result of Occupant's name: {confirmed_occupant_google_search_results}
Verification Analysis: {verification_analysis}

Then evaluate whether the occupant's business operations are reasonably aligned with B1 use based on standard land use interpretations in Singapore.

---

### FORMAT

Compliance level: <Unauthorised Use / Authorised Use / Likely Authorised Use / Likely Unauthorised Use / Need more information >
---End of Compliance---
Rationale: <Detailed rationale for compliance level>
---End of Rationale---
"""

                complianceprompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(compliance_rules),
                    HumanMessagePromptTemplate.from_template(compliance_prompt)
                ])

                compliance_chain = complianceprompt | llm

                try:
                    verified_compliance_response = compliance_chain.invoke({}).content.strip()
                    log_progress_realtime(f"✅ Compliance assessment completed", current_step)

                    # Parse compliance response using regex
                    compliance_match = re.search(r"(?i)\**Compliance Level:\s*(.*?)\s*-{3,}End of Compliance-{3,}", verified_compliance_response, re.DOTALL)
                    rationale_match = re.search(r"(?i)\**Rationale:\s*(.*?)\s*-{3,}End of Rationale-{3,}", verified_compliance_response, re.DOTALL)

                    compliance_level = compliance_match.group(1).strip() if compliance_match else "Assessment failed"
                    rationale = rationale_match.group(1).strip() if rationale_match else "Rationale not available"
                    
                except Exception as compliance_error:
                    log_progress_realtime(f"❌ Compliance assessment failed", current_step)
                    compliance_level = "Assessment failed"
                    rationale = f"Compliance assessment failed: {str(compliance_error)}"
            
            # Create result row with all 7 columns
            result = [
                address,                                    # address
                confirmed_occupant,                         # confirmed_occupant
                verification_analysis,                      # verification_analysis
                compliance_level,                           # compliance_level
                rationale,                                 # rationale
                address_search_results_raw if address_search_results_raw else "N/A",  # google_address_search_results
                confirmed_occupant_google_search_results if confirmed_occupant_google_search_results else "N/A"  # confirmed_occupant_google_search_results
            ]
            
            results.append(result)
            log_progress_realtime(f"✅ Processing completed for {address}", current_step)
            
        except Exception as e:
            error_msg = f"❌ Error processing {address}: {str(e)}"
            log_progress_realtime(error_msg, current_step)
            
            # Add error result to maintain structure
            results.append([
                address,
                "Error",
                str(e),
                "Unknown",
                f"Processing failed: {str(e)}",
                "N/A",
                "N/A"
            ])
    
    log_progress_realtime(f"🎉 Completed processing {len(results)} address(es)!")
    
    # Create CSV buffer
    csv_buffer = create_csv_for_download(results)
    
    return results, csv_buffer

def display_file_upload_page():
    """Display file upload page with address type selection."""
    # Display instruction at the top
    st.markdown("### Upload your file and then select the address type")
    
    # File uploader at the top
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv'],
        help="Upload your file containing addresses"
    )
    
    # Selectbox for address type
    address_type = st.selectbox(
        "Select the address type",
        options=["", "shophouse", "industrial"],
        help="Choose the type of addresses in your file"
    )
    
    # Display selected information
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
    
    if address_type:
        st.info(f"Selected address type: {address_type}")
    
    # Submit button
    submit_button = st.button("Submit", type="primary", use_container_width=True)
    
    # Process file when submitted
    if submit_button:
        if uploaded_file is None:
            st.error("Please upload a file first.")
        elif not address_type:
            st.error("Please select an address type.")
        else:
            # Initialize LLM (reuse from existing code)
            if 'llm' not in st.session_state:
                try:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model="gpt-4o",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        temperature=0
                    )
                    st.session_state.llm = llm
                except Exception as e:
                    st.error(f"Failed to initialize language model: {str(e)}")
                    return uploaded_file, address_type, submit_button
            
            # Process the selected address type
            st.markdown(f"### Processing {address_type.title()} Addresses")
            
            with st.spinner(f"Processing {address_type} addresses..."):
                result = process_csv_with_ui(uploaded_file, address_type, st.session_state.llm)
            
            if result:
                results, csv_buffer = result
                
                # Success message
                st.success(f"✅ Processing completed! Processed {len(results)} addresses.")
                
                # Download button
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{address_type}_processed_{timestamp}.csv"
                
                st.download_button(
                    label=f"📥 Download {address_type.title()} Results",
                    data=csv_buffer.getvalue(),
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
    
    return uploaded_file, address_type, submit_button

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
    mode = st.sidebar.radio("Select Mode:", options=["Compliance", "Enforcement"], index=1)

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
        uploaded_file, address_type, submit_button = display_file_upload_page()

if __name__ == "__main__":
    main()