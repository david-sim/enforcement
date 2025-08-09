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
    display_chat_interface
)

# Suppress warnings
warnings.filterwarnings("ignore", message="Importing verbose from langchain root module is no longer supported.*")

def main():
    """Main application entry point."""
    try:
        # Setup page configuration
        setup_page_config()
        
        # Display sidebar
        display_sidebar()
        
        # Main content area
        st.title("⚖️ Enforcement Processing Tool")
        
        # Check if there's conversation history to determine which interface to show
        if "history" not in st.session_state or len(st.session_state.history) <= 1:
            # Show file upload interface
            uploaded_file, address_type = display_file_upload_section()
            
            # Submit button
            submit_button = st.button("Submit", type="primary", use_container_width=True)
            
            # Process file when submitted
            if submit_button:
                if uploaded_file is None:
                    st.error("Please upload a file first.")
                elif not address_type:
                    st.error("Please select an address type.")
                else:
                    success, results = process_file_with_ui(uploaded_file, address_type)
                    if success:
                        st.balloons()
        else:
            # Show chat interface
            display_chat_interface()
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        with st.expander("Error Details", expanded=False):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
