"""
Main Streamlit application entry point.
Clean, focused on UI orchestration using modular components.
"""
import streamlit as st
import warnings
import traceback
from ui_components import (
    setup_page_config, 
    display_sidebar, 
    display_file_upload_section, 
    process_file_with_ui,
    process_single_record_with_ui,
    display_persistent_results,
    display_chat_interface
)
from about_page import display_about_page
from methodology_page import display_methodology_page
from login_handler import show_login_form, add_logout_button, is_authenticated

# Suppress warnings
warnings.filterwarnings("ignore", message="Importing verbose from langchain root module is no longer supported.*")

def check_authentication():
    """Check if user is authenticated and handle login flow."""
    # Check if user is authenticated (includes session timeout check)
    if not is_authenticated():
        show_login_form()
        return False
    
    return True

def display_main_page():
    """Display the main processing page."""
    # Main content area
    st.title("🛡️ Smart Compliance Operations Unit Tool")
    
    st.markdown("""
    Welcome to the AI-powered smart compliance operations system. You can either upload a CSV file 
    for bulk processing or enter a single record manually for quick analysis.
    """)
    
    # Initialize session state for persistent results
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'last_processed_inputs' not in st.session_state:
        st.session_state.last_processed_inputs = None
    
    # File upload and processing section (now includes manual entry option)
    uploaded_file, address_type, single_record_data = display_file_upload_section()
    
    # Create current input signature for change detection
    current_inputs = {
        'file_name': uploaded_file.name if uploaded_file else None,
        'file_size': uploaded_file.size if uploaded_file else None,
        'address_type': address_type,
        'single_record': single_record_data
    }
    
    # Check if inputs have changed (clear results if they have)
    if st.session_state.last_processed_inputs != current_inputs:
        if st.session_state.last_processed_inputs is not None:  # Don't clear on first load
            st.session_state.processing_results = None
            # st.info("🔄 Input changed - previous results cleared")
    
    # Determine processing mode and validate inputs
    processing_ready = False
    processing_mode = None
    
    if uploaded_file is not None and address_type:
        processing_ready = True
        processing_mode = "file"
    elif single_record_data is not None and address_type:
        processing_ready = True
        processing_mode = "single"
    
    # Show processing button if ready
    if processing_ready:
        if processing_mode == "file":
            button_label = "🚀 Start Processing CSV File"
            button_help = f"Process {uploaded_file.name} with {address_type} address type"
        else:
            button_label = "🚀 Process Single Record"
            button_help = f"Process the entered address as {address_type} type"
        
        if st.button(button_label, type="primary", use_container_width=True, 
                    help=button_help, key="start_processing_button"):
            # Clear progress messages for fresh processing run
            st.session_state.progress_messages = []
            # Flag to indicate we're starting processing (not showing results)
            st.session_state.show_persistent_results = False
            
            # Start processing immediately (don't use rerun)
            with st.spinner("Processing..."):
                if processing_mode == "file":
                    success, result_data = process_file_with_ui(uploaded_file, address_type)
                else:
                    success, result_data = process_single_record_with_ui(single_record_data, address_type)
                
                if success:
                    # Store results in session state and enable persistent display
                    st.session_state.processing_results = result_data
                    st.session_state.last_processed_inputs = current_inputs
                    st.session_state.show_persistent_results = True
                    st.balloons()
                else:
                    st.error("Processing failed. Please check your input and try again.")
    else:
        # Show help text for what's needed
        if address_type and uploaded_file is None and single_record_data is None:
            st.info("👆 Please either upload a CSV file or enter a single record manually to continue")
    
    # Display persistent results if available and flag is set
    if (st.session_state.processing_results is not None and 
        st.session_state.get('show_persistent_results', True)):  # Default to True for backward compatibility
        # Show persistent processing log first (in original location)
        if 'progress_messages' in st.session_state and st.session_state.progress_messages:
            st.markdown(f"### Processing {address_type.title()} Addresses")
            st.success(f"✅ Processing completed! Processed {len(st.session_state.processing_results[0])} addresses.")
            st.markdown("#### Processing Log")
            st.text_area(
                "Progress Log",
                value="\n".join(st.session_state.progress_messages),
                height=200,
                disabled=True,
                key=f"persistent_main_progress_log_{len(st.session_state.progress_messages)}"
            )
        
        st.markdown("---")
        display_persistent_results(st.session_state.processing_results, address_type)

def main():
    """Main application entry point."""
    try:
        # Setup page configuration ONLY ONCE
        if 'page_config_set' not in st.session_state:
            st.set_page_config(
                page_title="Smart Compliance Operations Unit Tool",
                page_icon="⚖️",
                layout="wide"
            )
            st.session_state.page_config_set = True
        
        # Check authentication first
        if not check_authentication():
            return  # Exit if not authenticated, login form is shown
        
        # Initialize session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Main'
        
        # Display sidebar (includes navigation and logout button)
        display_sidebar()
        add_logout_button()
        
        # Render the selected page
        current_page = st.session_state.current_page
        
        if current_page == 'About Us':
            display_about_page()
        elif current_page == 'Methodology':
            display_methodology_page()
        else:
            display_main_page()
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")
        
        # Display error details in expander for debugging
        with st.expander("Error Details (for debugging)"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
