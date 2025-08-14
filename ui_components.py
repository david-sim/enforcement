"""
UI utilities and Streamlit-specific components for the smart compliance operations application.
Contains reusable UI functions and progress handling.
"""
import streamlit as st
import datetime
import pytz
import pandas as pd
import io
import os
from typing import Tuple, Optional, Callable, List, Any
from enforcement_engine import process_addresses_batch


def load_sample_file(address_type: str) -> Optional[io.StringIO]:
    """
    Load sample CSV file based on address type.
    
    Args:
        address_type: Either "shophouse" or "industrial"
    
    Returns:
        StringIO object containing sample CSV data, or None if file not found
    """
    try:
        # Get the path to the sample file (relative to current working directory)
        sample_filename = f"{address_type}_sample_10_records.csv"
        sample_path = os.path.join("data", sample_filename)
        
        # Check if file exists
        if not os.path.exists(sample_path):
            st.error(f"❌ Sample file not found: {sample_path}")
            return None
        
        # Read the sample file
        with open(sample_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Create a StringIO object that mimics an uploaded file
        sample_file = io.StringIO(content)
        sample_file.name = sample_filename  # Add name attribute for compatibility
        sample_file.size = len(content)  # Add size attribute for compatibility
        
        return sample_file
        
    except Exception as e:
        st.error(f"❌ Error loading sample file: {str(e)}")
        return None


def display_file_upload_section() -> Tuple[Optional[Any], str, Optional[dict]]:
    """
    Display the file upload section with validation and manual entry option.
    Address type is selected first to enable future form customization.
    
    Returns:
        Tuple of (uploaded_file, address_type, single_record_data)
    """
    st.markdown("### Step 1: Select Address Type")
    
    # Address type selector
    address_type = st.selectbox(
        "Choose the type of address you're processing",
        options=["", "shophouse", "industrial"],
        help="Select address type first - this may customize the form fields below",
        key="address_type_selector"
    )
    
    if address_type:
        st.success(f"✅ Selected address type: **{address_type.title()}**")
        
        # Future customization point: different logic based on address_type
        # if address_type == "shophouse":
        #     # Shophouse-specific form customization
        # elif address_type == "industrial":
        #     # Industrial-specific form customization
    else:
        st.info("👆 Please select an address type to continue")
        return None, "", None
    
    # Show processing status if in progress
    st.markdown("### Step 2: Choose Input Method")
    
    # Input method selector
    input_method = st.radio(
        "How would you like to provide your data?",
        options=["Upload CSV file for multiple records/address", "Enter single record manually"],
        help="Choose between bulk processing with CSV or entering one record manually",
        key="input_method_selector"
    )
    
    # Clear validated record if switching input methods or address types
    if 'previous_input_method' not in st.session_state:
        st.session_state.previous_input_method = input_method
        st.session_state.previous_address_type = address_type
    elif (st.session_state.previous_input_method != input_method or 
          st.session_state.get('previous_address_type') != address_type):
        # Clear all related session state when user changes input method or address type
        st.session_state.validated_single_record = None
        st.session_state.progress_messages = []  # Clear progress log when user changes settings
        st.session_state.previous_input_method = input_method
        st.session_state.previous_address_type = address_type
    
    uploaded_file = None
    single_record_data = None
    
    # Initialize session state for single record data
    if 'validated_single_record' not in st.session_state:
        st.session_state.validated_single_record = None
    
    if input_method == "Upload CSV file for multiple records/address":
        st.markdown(f"#### 📁 Upload CSV File for {address_type.title()} Addresses")
        
        # Address-type-specific CSV format guidance
        with st.expander("📋 Expected CSV Format", expanded=False):
            st.markdown("**Required Columns:**")
            if address_type == "shophouse":
                st.code("""
Column 1: Address (Required) - e.g., "123 Smith Street #02-01 Singapore 123456"
Column 2: Primary Approved Use (Optional) - e.g., "Shophouse", "Commercial"  
Column 3: Secondary Approved Use (Optional) - e.g., "Retail", "Food & Beverage"
                """)
            else:  # industrial
                st.code("""
Column 1: Address (Required) - e.g., "1 Industrial Park Road Singapore 123456"
Column 2: Primary Approved Use (Optional) - e.g., "Industrial", "Warehouse"
Column 3: Secondary Approved Use (Optional) - e.g., "Manufacturing", "Storage"
                """)
        
        # Create two columns for file upload and sample data options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File uploader with static key
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help=f"Upload your CSV file containing {address_type} addresses",
                key="csv_file_uploader"
            )
        
        with col2:
            st.markdown("**Or use sample data:**")
            # Sample data button
            if st.button(
                f"📋 Load {address_type.title()} Sample",
                help=f"Load sample {address_type} addresses for testing (10 sample addresses)",
                key=f"load_sample_{address_type}",
                use_container_width=True
            ):
                # Load sample file
                sample_file = load_sample_file(address_type)
                if sample_file:
                    st.session_state['sample_file_loaded'] = sample_file
                    st.session_state['sample_file_type'] = address_type
                    st.success(f"✅ Loaded {address_type} sample data!")
                    st.rerun()
            
            # Show sample file info
            st.caption(f"💡 {address_type}_sample_10_records.csv")
        
        # Handle sample file loading
        sample_file = None
        if 'sample_file_loaded' in st.session_state and st.session_state.get('sample_file_type') == address_type:
            sample_file = st.session_state['sample_file_loaded']
            uploaded_file = sample_file  # Use sample file as uploaded file
            
            # Show sample data info with preview
            st.success(f"📋 Using {address_type} sample data with 10 sample addresses")
            
            # Preview the sample data
            with st.expander("👀 Preview Sample Data", expanded=False):
                try:
                    # Reset file pointer to beginning
                    sample_file.seek(0)
                    content = sample_file.read()
                    sample_file.seek(0)  # Reset again for later use
                    
                    # Show first few lines
                    lines = content.split('\n')[:4]  # Header + first 3 data rows
                    preview_content = '\n'.join(lines)
                    st.code(preview_content, language='csv')
                    
                    # Count total data rows (excluding header)
                    total_lines = len([line for line in content.split('\n') if line.strip()])
                    data_rows = total_lines - 1  # Subtract header
                    st.caption(f"Showing first 3 rows of {data_rows} total addresses")
                except Exception as e:
                    st.error(f"Error previewing sample data: {e}")
            
            # Clear sample button
            if st.button("❌ Clear Sample Data", key=f"clear_sample_{address_type}"):
                if 'sample_file_loaded' in st.session_state:
                    del st.session_state['sample_file_loaded']
                if 'sample_file_type' in st.session_state:
                    del st.session_state['sample_file_type']
                st.rerun()
        
        # Display uploaded file information
        elif uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Validate file size (optional)
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                st.warning("⚠️ File is large. Processing may take longer.")
    
    else:  # Manual entry
        st.markdown(f"#### ✍️ Enter Single {address_type.title()} Record")
        
        # Manual entry form with address-type-specific examples
        with st.form("manual_entry_form"):
            # Address-type-specific placeholder and help text
            if address_type == "shophouse":
                address_placeholder = "e.g., 123 Smith Street #02-01 Singapore 123456"
                address_help = "Enter the complete shophouse address including unit number and postal code"
                primary_placeholder = "e.g., Office, Restaurant"
                secondary_placeholder = "e.g., Eating House, Foodshop"
            else:  # industrial
                address_placeholder = "e.g., 1 Yishun Industrial Street 1 #01-05 768160"
                address_help = "Enter the complete industrial address including unit number and postal code"  
                primary_placeholder = "e.g., Manufacturing, Core Media"
                secondary_placeholder = "e.g., Minimart, Showroom"
            
            address = st.text_input(
                "Address *",
                placeholder=address_placeholder,
                help=address_help,
                key="manual_address"
            )
            
            # Primary Approved Use - mandatory for shophouse, optional for industrial
            if address_type == "shophouse":
                primary_approved_use = st.text_input(
                    "Primary Approved Use *",
                    placeholder=primary_placeholder,
                    help="Enter the primary approved use (required for shophouse)",
                    key="manual_primary_use"
                )
            else:  # industrial
                primary_approved_use = st.text_input(
                    "Primary Approved Use",
                    placeholder=primary_placeholder,
                    help="Enter the primary approved use (optional)",
                    key="manual_primary_use"
                )
            
            secondary_approved_use = st.text_input(
                "Secondary Approved Use", 
                placeholder=secondary_placeholder,
                help="Enter the secondary approved use (optional)",
                key="manual_secondary_use"
            )
            
            submitted = st.form_submit_button(
                "Validate Entry",
                type="secondary", 
                use_container_width=True
            )
            
            if submitted:
                # Validate address is required
                if not address.strip():
                    st.error("❌ Address is required!")
                    st.session_state.validated_single_record = None
                # Validate primary approved use for shophouse
                elif address_type == "shophouse" and not primary_approved_use.strip():
                    st.error("❌ Primary Approved Use is required for shophouse addresses!")
                    st.session_state.validated_single_record = None
                else:
                    # All validations passed
                    single_record_data = {
                        "address": address.strip(),
                        "primary_approved_use": primary_approved_use.strip() if primary_approved_use.strip() else None,
                        "secondary_approved_use": secondary_approved_use.strip() if secondary_approved_use.strip() else None
                    }
                    # Store validated data in session state
                    st.session_state.validated_single_record = single_record_data
                    st.success("✅ Record validated successfully!")
                    st.info(f"Address: {address}")
                    if primary_approved_use.strip():
                        st.info(f"Primary Use: {primary_approved_use}")
                    if secondary_approved_use.strip():
                        st.info(f"Secondary Use: {secondary_approved_use}")
        
        # Use session state data if available
        if st.session_state.validated_single_record is not None:
            single_record_data = st.session_state.validated_single_record
    
    # Debug information (temporary)
    with st.expander("Debug Info (click to expand)"):
        st.write(f"Address type: {address_type}")
        st.write(f"Input method: {input_method}")
        st.write(f"Uploaded file: {uploaded_file is not None}")
        st.write(f"Single record data: {single_record_data is not None}")
        if single_record_data:
            st.write(f"Record details: {single_record_data}")
    
    return uploaded_file, address_type, single_record_data


def create_realtime_progress_handler() -> Tuple[Callable, List[str]]:
    """Create progress handlers for real-time UI updates."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    progress_log = st.empty()
        
    # Use session state to persist progress messages
    if 'progress_messages' not in st.session_state:
        st.session_state.progress_messages = []
    progress_messages = st.session_state.progress_messages
    
    def update_progress(message, current_index=None, total=None):
        """Update progress with real-time UI updates."""
        # Use Singapore timezone for timestamps
        sg_tz = pytz.timezone("Asia/Singapore")
        timestamp = datetime.datetime.now(sg_tz).strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        # Always log to console
        print(formatted_message)

        # Add to progress messages
        progress_messages.append(formatted_message)
        st.session_state.progress_messages = progress_messages

        # Update progress bar if index provided
        if current_index is not None and total is not None:
            progress = current_index / total
            progress_bar.progress(progress)
            status_text.info(f"Processing {current_index}/{total}: {message}")

        # Show real-time progress log (newest messages at top)
        reversed_messages = list(reversed(progress_messages))
        display_messages = reversed_messages[:50]  # Limit to last 50 messages
        
        progress_log.text_area(
            "📋 Real-time Processing Log",
            value="\n".join(display_messages),
            height=200,
            disabled=True,
            key=f"realtime_progress_log_{len(progress_messages)}"
        )
    
    def clear_realtime_display():
        """Clear the real-time display elements to prevent duplication."""
        progress_bar.empty()
        status_text.empty()
        progress_log.empty()
    
    return update_progress, progress_messages, clear_realtime_display


def initialize_llm() -> Optional[Any]:
    """Initialize and cache the language model."""
    if 'llm' not in st.session_state:
        try:
            from langchain_openai import ChatOpenAI
            from config_manager import get_llm_model, get_llm_temperature
            
            # Validate API key exists (still from secrets for security)
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OPENAI_API_KEY not found in secrets")
                return None
            
            # Get model and temperature from config
            model = get_llm_model()
            temperature = float(get_llm_temperature())
            
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature
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
            
        st.success(f"✅ CSV validation passed! Found {len(addresses)} addresses to process")
        
    except ValueError as csv_error:
        error_msg = str(csv_error)
        st.error(f"❌ CSV Processing Error")
        
        # Check if it's a validation error and provide specific guidance
        if "CSV validation failed" in error_msg:
            st.error("**File Validation Issues:**")
            st.error(error_msg)
            
            # Provide specific guidance based on address type
            st.info("**Expected CSV Format:**")
            if address_type.lower() == "shophouse":
                st.code("""
Column 1: Address (Required) - e.g., "123 Smith Street #02-01 Singapore 123456"  
Column 2: Primary Approved Use (Optional) - e.g., "Shophouse"
Column 3: Secondary Approved Use (Optional) - e.g., "Retail"
""")
            elif address_type.lower() == "industrial":  
                st.code("""
Column 1: Address (Required) - e.g., "1 Industrial Park Road Singapore 123456"
Column 2: Primary Approved Use (Optional) - e.g., "Industrial"  
Column 3: Secondary Approved Use (Optional) - e.g., "Manufacturing"
""")
        else:
            st.error(error_msg)
            st.error("Please check that your file is a valid CSV with addresses in the first column.")
            
        return False, None
        
    except Exception as csv_error:
        st.error(f"❌ Unexpected error processing CSV file: {str(csv_error)}")
        st.error("Please check that your file is a valid CSV format.")
        return False, None
        
    # Initialize progress messages if not already present - CLEAR old messages for new processing
    if 'progress_messages' not in st.session_state:
        st.session_state.progress_messages = []
    else:
        # Clear previous progress messages when starting new processing
        st.session_state.progress_messages = []
    
    # Set processing state to disable UI elements
    st.session_state.processing_in_progress = True
    
    # Create progress handlers BEFORE the spinner so they're visible
    progress_callback, progress_messages, clear_realtime_display = create_realtime_progress_handler()

    # Process addresses with a simple spinner for the operation
    print(f"🔍 UI Debug: About to call process_addresses_batch with {len(addresses)} addresses")
    
    # Use a more lightweight progress indication approach
    with st.spinner(f"Processing {len(addresses)} {address_type} addresses..."):
        result = process_addresses_batch(addresses, llm, primary_approved_use_list, secondary_approved_use_list, address_type, progress_callback)
    
    print(f"🔍 UI Debug: Got result of type: {type(result)}")

    # Check and process the result
    if result is None:
        st.error("❌ Processing function returned None - this shouldn't happen!")
        print("❌ UI Debug: result is None!")
        clear_realtime_display()  # Clear real-time displays before re-enabling UI
        st.session_state.processing_in_progress = False  # Re-enable UI on error
        return False, None

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        st.error(f"❌ Processing function returned unexpected format: {type(result)}")
        print(f"❌ UI Debug: result format error - type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'no len'}")
        clear_realtime_display()  # Clear real-time displays before re-enabling UI
        st.session_state.processing_in_progress = False  # Re-enable UI on error
        return False, None

    try:
        print(f"🔍 UI Debug: About to unpack result: {type(result)} with length {len(result)}")
        results, csv_buffer = result
        print(f"🔍 UI Debug: Successfully unpacked - results: {type(results)}, csv_buffer: {type(csv_buffer)}")

    except Exception as processing_error:
        st.error(f"❌ Error during address processing: {str(processing_error)}")
        print(f"❌ UI Debug: Exception caught: {str(processing_error)}")
        import traceback
        traceback.print_exc()
        clear_realtime_display()  # Clear real-time displays before re-enabling UI
        st.session_state.processing_in_progress = False  # Re-enable UI on error
        return False, None

    # Processing completed successfully - clear real-time displays and re-enable UI
    clear_realtime_display()  # Clear progress bar, status, and real-time log to prevent duplication
    st.session_state.processing_in_progress = False

    # Success message will be handled by main application
    return True, (results, csv_buffer)


def process_single_record_with_ui(single_record_data: dict, address_type: str) -> Tuple[bool, Optional[Any]]:
    """
    Process a single manually entered record with real-time UI updates.
    
    Args:
        single_record_data: Dictionary containing address, primary_approved_use, secondary_approved_use
        address_type: Selected address type
    
    Returns:
        Success status and results
    """
    # Initialize LLM
    llm = initialize_llm()
    if llm is None:
        return False, None
    
    # Prepare data for processing
    addresses = [single_record_data["address"]]
    primary_approved_use_list = [single_record_data.get("primary_approved_use")] if single_record_data.get("primary_approved_use") else [None]
    secondary_approved_use_list = [single_record_data.get("secondary_approved_use")] if single_record_data.get("secondary_approved_use") else [None]
    
    st.success(f"✅ Processing single {address_type} address")
    
    # Clear previous progress messages when starting new processing
    st.session_state.progress_messages = []
    
    # Set processing state to disable UI elements
    st.session_state.processing_in_progress = True
    
    # Create progress handlers BEFORE the spinner so they're visible
    progress_callback, progress_messages, clear_realtime_display = create_realtime_progress_handler()
    
    # Process the single address with lightweight spinner
    print(f"🔍 UI Debug: About to call process_addresses_batch with 1 address")
    
    with st.spinner(f"Processing 1 {address_type} address..."):
        result = process_addresses_batch(addresses, llm, primary_approved_use_list, secondary_approved_use_list, address_type, progress_callback)
    
    print(f"🔍 UI Debug: Got result of type: {type(result)}")
            
    # Debug: Check what we got back
    if result is None:
        st.error("❌ Processing function returned None - this shouldn't happen!")
        print("❌ UI Debug: result is None!")
        clear_realtime_display()  # Clear real-time displays before re-enabling UI
        st.session_state.processing_in_progress = False  # Re-enable UI on error
        return False, None
    
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        st.error(f"❌ Processing function returned unexpected format: {type(result)}")
        print(f"❌ UI Debug: result format error - type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'no len'}")
        clear_realtime_display()  # Clear real-time displays before re-enabling UI
        st.session_state.processing_in_progress = False  # Re-enable UI on error
        return False, None
    
    try:
        # Correct unpacking: process_addresses_batch returns (results_list, csv_buffer)
        results, csv_buffer = result
        print(f"🔍 UI Debug: results type={type(results)}, csv_buffer type={type(csv_buffer)}")
        
        if not results:
            st.warning("⚠️ Processing completed but no results returned")
            print("⚠️ UI Debug: No results returned")
            clear_realtime_display()  # Clear real-time displays before re-enabling UI
            st.session_state.processing_in_progress = False  # Re-enable UI
            return False, None
        
        print(f"🔍 UI Debug: Results length: {len(results)}")
        
        # Processing completed successfully - clear real-time displays and re-enable UI
        clear_realtime_display()  # Clear progress bar, status, and real-time log to prevent duplication
        st.session_state.processing_in_progress = False
        
        # Success message will be handled by main application
        return True, (results, csv_buffer)
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        print(f"❌ UI Debug: Exception during processing: {str(e)}")
        import traceback
        print(f"❌ UI Debug: Full traceback: {traceback.format_exc()}")
        clear_realtime_display()  # Clear real-time displays before re-enabling UI
        st.session_state.processing_in_progress = False  # Re-enable UI on error
        return False, None


def generate_summary_report(results: List[List[str]], address_type: str) -> bytes:
    """
    Generate a text summary report of the processing results.
    
    Args:
        results: List of processed address results
        address_type: Type of addresses processed
    
    Returns:
        Text report content as bytes
    """
    return generate_text_summary_report(results, address_type)


def generate_text_summary_report(results: List[List[str]], address_type: str) -> bytes:
    """
    Generate a simple text-based summary report.
    """
    buffer = io.StringIO()
    
    sg_tz = pytz.timezone("Asia/Singapore")
    current_time = datetime.datetime.now(sg_tz).strftime("%Y-%m-%d %H:%M:%S SGT")
    
    buffer.write("SMART COMPLIANCE OPERATIONS SUMMARY REPORT\n")
    buffer.write("=" * 50 + "\n\n")
    buffer.write(f"Report Generated: {current_time}\n")
    buffer.write(f"Address Type: {address_type.title()}\n")
    buffer.write(f"Total Addresses Processed: {len(results)}\n\n")
    
    if results:
        columns = [
            'Address', 'Confirmed Occupant', 'Verification Analysis', 'Primary Approved Use',
            'Secondary Approved Use', 'Compliance Level', 'Rationale', 'Google Address Search',
            'Google Address Search Variant', 'Occupant Google Search'
        ]
        df = pd.DataFrame(results, columns=columns)
        
        # Key Metrics
        buffer.write("KEY METRICS\n")
        buffer.write("-" * 20 + "\n")
        
        total_addresses = len(results)
        successful_occupants = df[~df['Confirmed Occupant'].isin(['Need more information', 'Error'])].shape[0]
        completed_assessments = df[~df['Compliance Level'].isin(['Need more information', 'Assessment failed', 'Error'])].shape[0]
        error_count = df[df['Confirmed Occupant'] == 'Error'].shape[0]
        
        occupant_rate = (successful_occupants/total_addresses)*100 if total_addresses > 0 else 0
        assessment_rate = (completed_assessments/total_addresses)*100 if total_addresses > 0 else 0
        error_rate = (error_count/total_addresses)*100 if total_addresses > 0 else 0
        
        buffer.write(f"Occupant Successfully Identified: {successful_occupants}/{total_addresses} ({occupant_rate:.1f}%)\n")
        buffer.write(f"Compliance Assessed: {completed_assessments}/{total_addresses} ({assessment_rate:.1f}%)\n")
        buffer.write(f"Processing Errors: {error_count}/{total_addresses} ({error_rate:.1f}%)\n\n")
        
        # Compliance Distribution
        buffer.write("COMPLIANCE LEVEL DISTRIBUTION\n")
        buffer.write("-" * 30 + "\n")
        
        compliance_counts = df['Compliance Level'].value_counts()
        for level, count in compliance_counts.items():
            percentage = (count / total_addresses) * 100
            buffer.write(f"{level}: {count} ({percentage:.1f}%)\n")
        
        buffer.write("\n")
        
        # Key Insights
        buffer.write("KEY INSIGHTS\n")
        buffer.write("-" * 15 + "\n")
        
        if not compliance_counts.empty:
            most_common = compliance_counts.index[0]
            most_common_count = compliance_counts.iloc[0]
            buffer.write(f"• Most common compliance level: {most_common} ({most_common_count} addresses, {(most_common_count/total_addresses)*100:.1f}%)\n")
        
        unauthorized_count = compliance_counts.get('Unauthorised Use', 0) + compliance_counts.get('Likely Unauthorised Use', 0)
        authorized_count = compliance_counts.get('Authorised Use', 0) + compliance_counts.get('Likely Authorised Use', 0)
        
        if unauthorized_count > 0 or authorized_count > 0:
            buffer.write(f"• Potential compliance issues: {unauthorized_count} addresses ({(unauthorized_count/total_addresses)*100:.1f}%)\n")
            buffer.write(f"• Compliant addresses: {authorized_count} addresses ({(authorized_count/total_addresses)*100:.1f}%)\n")
        
        need_info_count = df[df['Confirmed Occupant'] == 'Need more information'].shape[0]
        if need_info_count > 0:
            buffer.write(f"• Addresses needing additional information: {need_info_count} ({(need_info_count/total_addresses)*100:.1f}%)\n")
        
        buffer.write(f"\n")
        buffer.write("=" * 50 + "\n")
        buffer.write("End of Report\n")
    
    else:
        buffer.write("No results to summarize.\n")
    
    # Convert string to bytes with proper encoding
    text_content = buffer.getvalue()
    buffer.close()
    return text_content.encode('utf-8')


def display_single_record_details(record_data: List[str], address_type: str):
    """
    Display detailed results for a single record in an easy-to-read format.
    
    Args:
        record_data: Single record result list
        address_type: Type of address processed
    """
    st.markdown("### 🏠 Record Details")
    
    # Parse the record data based on the expected format
    # Expected format: [address, confirmed_occupant, verification_analysis, 
    #                  primary_approved_use, secondary_approved_use, compliance_level, 
    #                  rationale, google_address_search_results, 
    #                  google_address_search_results_variant, confirmed_occupant_google_search_results]
    
    if len(record_data) >= 10:
        address = record_data[0]
        confirmed_occupant = record_data[1]
        verification_analysis = record_data[2]
        primary_approved_use = record_data[3]
        secondary_approved_use = record_data[4]
        compliance_level = record_data[5]
        rationale = record_data[6]
        google_address_search = record_data[7]
        google_address_variant = record_data[8]
        occupant_google_search = record_data[9]
        
        # Single column layout with address information first, then occupant information
        st.markdown("#### 📍 Address Information")
        st.info(f"**Address:** {address}")
        
        if primary_approved_use and primary_approved_use.strip():
            st.info(f"**Primary Approved Use:** {primary_approved_use}")
        
        if secondary_approved_use and secondary_approved_use.strip():
            st.info(f"**Secondary Approved Use:** {secondary_approved_use}")
        
        st.markdown("#### 🔍 Occupant Information")
        st.info(f"**Identified Occupant:** {confirmed_occupant}")
        
        # Verification Analysis
        if verification_analysis and verification_analysis.strip() and verification_analysis.lower() != 'n/a':
            with st.expander("🔬 Verification Analysis", expanded=True):
                st.write(verification_analysis)
        
        # Compliance Assessment section (full width, before rationale)
        st.markdown("#### ⚖️ Compliance Assessment")
        if compliance_level.lower() in ['compliant', 'low risk']:
            st.success(f"**Status:** {compliance_level}")
        elif compliance_level.lower() in ['medium risk', 'moderate']:
            st.warning(f"**Status:** {compliance_level}")
        elif compliance_level.lower() in ['high risk', 'non-compliant']:
            st.error(f"**Status:** {compliance_level}")
        else:
            st.info(f"**Status:** {compliance_level}")
        
        # Rationale section (full width)
        if rationale and rationale.strip() and rationale.lower() != 'n/a':
            st.markdown("#### 📝 Analysis Rationale")
            with st.expander("View Detailed Rationale", expanded=True):
                st.write(rationale)
    
    else:
        st.error("⚠️ Record data format is incomplete. Please check the processing results.")
        st.write(f"Expected 10 fields, got {len(record_data)} fields")
        if record_data:
            st.write("Available data:", record_data)


def display_persistent_results(result_data: Tuple, address_type: str):
    """
    Display persistent results that survive page reruns.
    
    Args:
        result_data: Tuple containing (results, csv_buffer) 
        address_type: Type of addresses processed
    """
    if result_data is None:
        return
        
    results, csv_buffer = result_data
    
    st.markdown("## 📋 Processing Results")
    # Success message is already displayed in main application
    
    # For single records, display detailed results immediately
    if len(results) == 1:
        display_single_record_details(results[0], address_type)
    
    # Download buttons section
    st.markdown("### 📥 Downloads")
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate timestamp for filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{address_type}_results_{timestamp}.csv"
        
        # CSV download button
        st.download_button(
            label=f"📥 Download {address_type.title()} Results",
            data=csv_buffer.getvalue(),
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
            key=f"download_csv_{timestamp}"  # Unique key to prevent widget conflicts
        )
    
    with col2:
        # Generate text summary report
        try:
            text_buffer = generate_summary_report(results, address_type)
            report_filename = f"{address_type}_summary_report_{timestamp}.txt"
            
            st.download_button(
                label=f"📑 Download Summary Report",
                data=text_buffer,
                file_name=report_filename,
                mime="text/plain",
                use_container_width=True,
                key=f"download_report_{timestamp}"  # Unique key to prevent widget conflicts
            )
        except Exception as e:
            st.warning(f"Report generation unavailable: {str(e)}")
    
    # Display results summary
    display_results_summary(results, address_type)


def display_results_summary(results: List[List[str]], address_type: str):
    """
    Display overall summary of processing results with statistics and visualizations.
    
    Args:
        results: List of processed address results
        address_type: Type of addresses processed (shophouse/industrial)
    """
    # Section header
    st.markdown("## 📊 Overall Summary")
    
    if not results:
        st.warning("No results to summarize.")
        return
    
    # Convert results to DataFrame for easier analysis
    # Based on enforcement_engine.py, results structure is:
    # [address, confirmed_occupant, verification_analysis, primary_approved_use, 
    #  secondary_approved_use, compliance_level, rationale, google_address_search_results,
    #  google_address_search_results_variant, confirmed_occupant_google_search_results]
    
    columns = [
        'Address', 'Confirmed Occupant', 'Verification Analysis', 'Primary Approved Use',
        'Secondary Approved Use', 'Compliance Level', 'Rationale', 'Google Address Search',
        'Google Address Search Variant', 'Occupant Google Search'
    ]
    
    df = pd.DataFrame(results, columns=columns)
    
    # Basic statistics
    total_addresses = len(results)
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📍 Total Addresses Processed",
            value=total_addresses
        )
    
    with col2:
        # Count successful occupant identifications
        successful_occupants = df[~df['Confirmed Occupant'].isin(['Need more information', 'Error'])].shape[0]
        occupant_success_rate = (successful_occupants / total_addresses) * 100 if total_addresses > 0 else 0
        st.metric(
            label="✅ Occupant ID Success Rate",
            value=f"{occupant_success_rate:.1f}%",
            delta=f"{successful_occupants}/{total_addresses}"
        )
    
    with col3:
        # Count compliance assessments (not "Need more information" or "Assessment failed")
        completed_assessments = df[~df['Compliance Level'].isin(['Need more information', 'Assessment failed', 'Error'])].shape[0]
        assessment_rate = (completed_assessments / total_addresses) * 100 if total_addresses > 0 else 0
        st.metric(
            label="⚖️ Compliance Assessed",
            value=f"{assessment_rate:.1f}%",
            delta=f"{completed_assessments}/{total_addresses}"
        )
    
    with col4:
        # Processing errors
        error_count = df[df['Confirmed Occupant'] == 'Error'].shape[0]
        error_rate = (error_count / total_addresses) * 100 if total_addresses > 0 else 0
        st.metric(
            label="❌ Processing Errors",
            value=f"{error_rate:.1f}%",
            delta=f"{error_count}/{total_addresses}"
        )
    
    # Compliance Level Distribution
    st.markdown("### 🎯 Compliance Level Distribution")
    
    compliance_counts = df['Compliance Level'].value_counts()
    
    if not compliance_counts.empty:
        # Create two columns for pie chart and table
        chart_col, table_col = st.columns([2, 1])
        
        with chart_col:
            # Create bar chart using Streamlit's built-in chart
            st.subheader("Distribution Chart")
            
            # Create bar chart data
            chart_data = compliance_counts.reset_index()
            chart_data.columns = ['Compliance Level', 'Count']
            
            st.bar_chart(data=chart_data.set_index('Compliance Level'))
        
        with table_col:
            st.subheader("Summary Table")
            
            # Create summary table with percentages
            summary_df = pd.DataFrame({
                'Compliance Level': compliance_counts.index,
                'Count': compliance_counts.values,
                'Percentage': (compliance_counts.values / total_addresses * 100).round(1)
            })
            
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )
    
    # Additional insights
    st.markdown("### 💡 Enforcement Insights")
    
    insights = []
    
    # Most common compliance level
    if not compliance_counts.empty:
        most_common = compliance_counts.index[0]
        most_common_count = compliance_counts.iloc[0]
        insights.append(f"🔸 Most common compliance level: **{most_common}** ({most_common_count} addresses, {(most_common_count/total_addresses)*100:.1f}%)")
    
    # Unauthorized vs Authorized breakdown
    unauthorized_count = compliance_counts.get('Unauthorised Use', 0) + compliance_counts.get('Likely Unauthorised Use', 0)
    authorized_count = compliance_counts.get('Authorised Use', 0) + compliance_counts.get('Likely Authorised Use', 0)
    
    if unauthorized_count > 0 or authorized_count > 0:
        insights.append(f"🔸 Potential compliance issues: **{unauthorized_count}** addresses ({(unauthorized_count/total_addresses)*100:.1f}%)")
        insights.append(f"🔸 Compliant addresses: **{authorized_count}** addresses ({(authorized_count/total_addresses)*100:.1f}%)")
    
    # Occupant identification insights
    need_info_count = df[df['Confirmed Occupant'] == 'Need more information'].shape[0]
    if need_info_count > 0:
        insights.append(f"🔸 Addresses needing additional information: **{need_info_count}** ({(need_info_count/total_addresses)*100:.1f}%)")
    
    # Display insights
    for insight in insights:
        st.markdown(insight)
    
    # Address type confirmation
    st.markdown(f"📋 **Address Type Processed:** {address_type.title()}")
    
    # Processing timestamp
    sg_tz = pytz.timezone("Asia/Singapore")
    current_time = datetime.datetime.now(sg_tz).strftime("%Y-%m-%d %H:%M:%S SGT")
    st.markdown(f"🕒 **Summary Generated:** {current_time}")


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
    """Display the sidebar with application information and navigation."""
    with st.sidebar:
        # Load and display main icon with transparent background
        try:
            from PIL import Image
            main_icon = Image.open("imgs/icon_transparent-color-nobg.png")
            st.image(main_icon, use_container_width=True)
        except:
            st.info("Main icon not found")
        
        st.markdown("---")
        
        # Navigation using radio buttons (Streamlit best practice)
        st.markdown("### 🧭 Navigation")
        
        # Initialize session state for page navigation
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Main'
        
        # Radio button navigation
        page_options = ['Main', 'About Us', 'Methodology']
        current_index = page_options.index(st.session_state.current_page)
        
        selected_page = st.radio(
            "Select Page:",
            options=page_options,
            index=current_index,
            key="page_navigation_radio_unique"
        )
        
        # Update session state if page changed
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats or info
        st.markdown("### 📊 Quick Info")
        st.info("💡 Upload CSV files with addresses to get started")
        st.success("🤖 AI-assisted for accurate results")
        
        st.markdown("---")
        st.markdown("**Smart Compliance Operations Unit Tool**")
        st.markdown("Upload CSV files to process addresses for compliance assessment.")
        
        
        # Powered By label and Streamlit sidebar image
        st.markdown("---")
        st.markdown("**Powered By**")
        try:
            sidebar_image = Image.open("imgs/stsidebarimg.png")
            st.image(sidebar_image, use_container_width=True)
        except:
            st.info("Sidebar image not found")


def setup_page_config():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="Smart Compliance Operations Unit Tool",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Load custom CSS or styling if needed
    # st.markdown("<style>...</style>", unsafe_allow_html=True)
