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
from about_page import display_about_page
from methodology_page import display_methodology_page

# Suppress warnings
warnings.filterwarnings("ignore", message="Importing verbose from langchain root module is no longer supported.*")

def display_main_page():
    """Display the main processing page."""
    # Main content area
    st.title("⚖️ Enforcement Processing Tool")
    
    st.markdown("""
    Welcome to the AI-powered enforcement processing system. Upload your CSV file 
    containing addresses and approved use information to begin automated processing.
    """)
    
    # File upload and processing section
    uploaded_file, address_type = display_file_upload_section()
    
    # Process file if both file and address type are provided
    if uploaded_file is not None and address_type:
        if st.button("🚀 Start Processing", type="primary", use_container_width=True, 
                    key="start_processing_button"):
            with st.spinner("Processing addresses..."):
                success, result_data = process_file_with_ui(uploaded_file, address_type)
                
                if success:
                    st.balloons()
                else:
                    st.error("Processing failed. Please check your file and try again.")

def main():
    """Main application entry point."""
    try:
        # Setup page configuration
        setup_page_config()
        
        # Initialize session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Main'
        
        # Display sidebar (includes navigation)
        display_sidebar()
        
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
