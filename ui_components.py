"""
UI utilities and Streamlit-specific components for the enforcement application.
Contains reusable UI functions and progress handling.
"""
import streamlit as st
import datetime
from typing import Tuple, Optional, Callable, List, Any
from enforcement_engine import process_addresses_batch


def display_file_upload_section() -> Tuple[Optional[Any], str]:
    """
    Display the file upload section with validation.
    
    Returns:
        Tuple of (uploaded_file, address_type)
    """
    st.markdown("### Upload your file and then select the address type")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv'],
        help="Upload your file containing addresses"
    )
    
    # Address type selector
    address_type = st.selectbox(
        "Select the address type",
        options=["", "shophouse", "industrial"],
        help="Choose the type of addresses in your file"
    )
    
    # Display selected information
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Validate file size (optional)
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
            st.warning("⚠️ File is large. Processing may take longer.")
    
    if address_type:
        st.info(f"Selected address type: {address_type}")
    
    return uploaded_file, address_type


def create_realtime_progress_handler() -> Tuple[Callable, List[str]]:
    """Create progress handlers for real-time UI updates."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    progress_log = st.empty()
    progress_messages = []
    
    def update_progress(message, current_index=None, total=None):
        """Update progress with real-time UI updates."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Always log to console
        print(formatted_message)
        
        # Add to progress messages
        progress_messages.append(formatted_message)
        
        # Update progress bar if index provided
        if current_index is not None and total is not None:
            progress = current_index / total
            progress_bar.progress(progress)
            status_text.info(f"Processing {current_index}/{total}: {message}")
        
        # Update progress log with scrollable text area
        progress_log.text_area(
            "Processing Log",
            value="\n".join(progress_messages),
            height=200,
            disabled=True,
            key=f"progress_log_{len(progress_messages)}"
        )
    
    return update_progress, progress_messages


def initialize_llm() -> Optional[Any]:
    """Initialize and cache the language model."""
    if 'llm' not in st.session_state:
        try:
            from langchain_openai import ChatOpenAI
            
            # Validate API key exists
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OPENAI_API_KEY not found in secrets")
                return None
            
            llm = ChatOpenAI(
                model="gpt-4o",
                api_key=api_key,
                temperature=0
            )
            st.session_state.llm = llm
            return llm
        except Exception as e:
            st.error(f"Failed to initialize language model: {str(e)}")
            return None
    return st.session_state.llm


def process_file_with_ui(uploaded_file: Any, address_type: str) -> Tuple[bool, Optional[Any]]:
    """
    Process uploaded file with real-time UI updates.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        address_type: Selected address type
    
    Returns:
        Success status and results
    """
    # Initialize LLM
    llm = initialize_llm()
    if llm is None:
        return False, None
    
    # Process CSV to extract addresses
    try:
        from enforcement_engine import process_csv
        csv_data = process_csv(address_type, uploaded_file)
        addresses = csv_data.get("addresses", [])
        primary_approved_use_list = csv_data.get("primary_approved_use", [])
        secondary_approved_use_list = csv_data.get("secondary_approved_use", [])
        
        if not addresses:
            st.error("No addresses found in the uploaded file.")
            return False, None
            
        st.info(f"Found {len(addresses)} addresses to process")
        
    except Exception as csv_error:
        st.error(f"❌ Error processing CSV file: {str(csv_error)}")
        st.error("Please check that your file is a valid CSV with addresses in the first column.")
        return False, None
        
    # Create progress handlers
    progress_callback, progress_messages = create_realtime_progress_handler()
    
    # Process addresses
    st.markdown(f"### Processing {address_type.title()} Addresses")
    
    with st.spinner(f"Processing {address_type} addresses..."):
            try:
                print(f"🔍 UI Debug: About to call process_addresses_batch with {len(addresses)} addresses")
                result = process_addresses_batch(addresses, llm, primary_approved_use_list, secondary_approved_use_list, address_type, progress_callback)
                print(f"🔍 UI Debug: Got result of type: {type(result)}")
                
                # Debug: Check what we got back
                if result is None:
                    st.error("❌ Processing function returned None - this shouldn't happen!")
                    print("❌ UI Debug: result is None!")
                    return False, None
                
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    st.error(f"❌ Processing function returned unexpected format: {type(result)}")
                    print(f"❌ UI Debug: result format error - type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'no len'}")
                    return False, None
                
                print(f"🔍 UI Debug: About to unpack result: {type(result)} with length {len(result)}")
                results, csv_buffer = result
                print(f"🔍 UI Debug: Successfully unpacked - results: {type(results)}, csv_buffer: {type(csv_buffer)}")
                
            except Exception as processing_error:
                st.error(f"❌ Error during address processing: {str(processing_error)}")
                print(f"❌ UI Debug: Exception caught: {str(processing_error)}")
                import traceback
                traceback.print_exc()
                return False, None
    
    # Display success message
    st.success(f"✅ Processing completed! Processed {len(results)} addresses.")
    
    # Create download button
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{address_type}_processed_{timestamp}.csv"
    
    st.download_button(
        label=f"📥 Download {address_type.title()} Results",
        data=csv_buffer.getvalue(),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )
    
    return True, (results, csv_buffer)


def display_chat_interface():
    """Display the chat interface for conversational interactions."""
    # Initialize conversation history
    if "history" not in st.session_state:
        st.session_state.history = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]
    
    # Display conversation history
    NUMBER_OF_MESSAGES_TO_DISPLAY = 20
    for message in st.session_state.history[-NUMBER_OF_MESSAGES_TO_DISPLAY:]:
        role = message["role"]
        avatar_image = "imgs/avatar.png" if role == "assistant" else "imgs/stuser.png" if role == "user" else None
        with st.chat_message(role, avatar=avatar_image):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history
        st.session_state.history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="imgs/stuser.png"):
            st.write(prompt)
        
        # TODO: Add chat processing logic here
        # For now, just echo back
        with st.chat_message("assistant", avatar="imgs/avatar.png"):
            response = f"I received your message: {prompt}"
            st.write(response)
            st.session_state.history.append({"role": "assistant", "content": response})


def display_sidebar():
    """Display the sidebar with application information."""
    with st.sidebar:
        # Load and display sidebar image
        try:
            from PIL import Image
            sidebar_image = Image.open("imgs/stsidebarimg.png")
            st.image(sidebar_image, use_container_width=True)
        except:
            st.info("Sidebar image not found")
        
        st.markdown("---")
        st.markdown("**Enforcement Processing Tool**")
        st.markdown("Upload CSV files to process addresses for compliance assessment.")


def setup_page_config():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="Enforcement Tool",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Load custom CSS or styling if needed
    # st.markdown("<style>...</style>", unsafe_allow_html=True)
