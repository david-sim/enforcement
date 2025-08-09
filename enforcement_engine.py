"""
Core enforcement processing engine.
Contains the main address processing logic without UI dependencies.
"""
import datetime
import re
import pandas as pd
import io
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from search_service import google_search_entity
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


def get_occupant_rules() -> str:
    """Get occupant rules from secrets, with fallback for testing"""
    try:
        import streamlit as st
        return st.secrets.get("INDUSTRIAL_OCCUPANT_RULES", "")
    except:
        return ""


def get_compliance_rules() -> str:
    """Get compliance rules from secrets, with fallback for testing"""
    try:
        import streamlit as st
        return st.secrets.get("INDUSTRIAL_COMPLIANCE_RULES", "")
    except:
        return ""


def process_csv(command: str, csv_file) -> Dict[str, List[str]]:
    """
    Process the uploaded CSV file based on the command ("shophouse" or "industrial").

    Args:
        command: The command, either "shophouse" or "industrial"
        csv_file: The uploaded CSV file object

    Returns:
        Dictionary containing the processed data
    """
    csv_file.seek(0)
    
    # Try to detect if the CSV uses semicolon separator
    # Handle both string and bytes file objects
    try:
        first_line = csv_file.readline()
        # If it's bytes, decode it
        if isinstance(first_line, bytes):
            first_line = first_line.decode('utf-8')
        first_line = first_line.strip()
        csv_file.seek(0)
    except Exception as e:
        print(f"⚠️ Error reading first line: {e}")
        first_line = ""
        csv_file.seek(0)
    
    # Check if the first line contains semicolons (might be semicolon-separated)
    try:
        if first_line and ';' in first_line and ',' not in first_line:
            print("🔍 Detected semicolon-separated CSV")
            df = pd.read_csv(csv_file, sep=';', header=0)
        else:
            print("🔍 Using standard comma-separated CSV")
            df = pd.read_csv(csv_file, header=0)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        # Try alternative encoding or approach
        csv_file.seek(0)
        try:
            df = pd.read_csv(csv_file, header=0, encoding='utf-8')
            print("✅ Successfully read CSV with utf-8 encoding")
        except Exception as e2:
            print(f"❌ Failed with utf-8 too: {e2}")
            raise ValueError(f"Could not read CSV file: {e2}")
    
    # Debug: Print DataFrame structure
    print(f"📊 CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    if len(df) > 0:
        print(f"   First address: '{df.iloc[0, 0]}'")
    print("---")

    if command.lower() == "shophouse":
        addresses = df.iloc[:, 0].dropna().tolist()
        primary_approved_use = df.iloc[:, 1].dropna().tolist() if df.shape[1] > 1 else []
        secondary_approved_use = df.iloc[:, 2].dropna().tolist() if df.shape[1] > 2 else []
        
        # Debug: Print extracted data
        print(f"📍 Extracted {len(addresses)} addresses for {command}")
            
        return {
            "addresses": addresses,
            "primary_approved_use": primary_approved_use,
            "secondary_approved_use": secondary_approved_use
        }
    elif command.lower() == "industrial":
        addresses = df.iloc[:, 0].dropna().tolist()
        primary_approved_use = df.iloc[:, 1].dropna().tolist() if df.shape[1] > 1 else []
        secondary_approved_use = df.iloc[:, 2].dropna().tolist() if df.shape[1] > 2 else []
        
        # Debug: Print extracted data  
        print(f"📍 Extracted {len(addresses)} addresses for {command}")
            
        return {
            "addresses": addresses,
            "primary_approved_use": primary_approved_use,
            "secondary_approved_use": secondary_approved_use
        }
    else:
        raise ValueError("Invalid command. Must be 'shophouse' or 'industrial'.")


def create_csv_for_download(results_data: List[List[str]]) -> io.BytesIO:
    """
    Create a CSV file buffer for download with the specified columns.
    
    Args:
        results_data: List of result rows with all required columns
    
    Returns:
        BytesIO buffer containing the CSV data
    """
    columns = [
        'address',
        'confirmed_occupant', 
        'verification_analysis',
        'compliance_level',
        'rationale',
        'google_address_search_results',
        'google_address_search_results_variant',
        'confirmed_occupant_google_search_results'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(results_data, columns=columns)
    
    # Create CSV buffer
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)
    
    return csv_buffer


def process_single_address(address: str, llm: Any, primary_approved_use: str = "", secondary_approved_use: str = "", progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    """
    Process a single address through the complete enforcement workflow.
    
    Args:
        address: Address to process
        llm: Language model instance
        primary_approved_use: Primary approved use for the address
        secondary_approved_use: Secondary approved use for the address
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Processing result with all required fields
    """
    def log_progress(message: str) -> None:
        if progress_callback:
            progress_callback(message)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

    log_progress(f"📍 Processing {address}")
    
    # Clean the address - remove any extra data that might be semicolon-separated
    clean_address = address.split(';')[0].strip() if ';' in address else address.strip()
    
    if clean_address != address:
        log_progress(f"🧹 Cleaned address: '{address}' → '{clean_address}'")
    
    # Initialize variables
    address_search_results_raw = ""
    address_search_results_raw_variant = ""
    confirmed_occupant_google_search_results = ""
    verified_occupant_response = ""
    verification_analysis = ""
    confirmed_occupant = ""
    compliance_level = "Need more information"
    rationale = "Unable to confirm occupant, compliance assessment not performed."
    
    # Get system rules
    occupant_rules = get_occupant_rules()
    compliance_rules = get_compliance_rules()
    today = datetime.date.today().isoformat()
    
    # Step 1: Google search for address
    log_progress(f"🔍 Searching for address information...")
    
    try:
        # Original address search - use cleaned address
        address_search_query = f"{clean_address}"
        
        # Debug: Show what we're actually searching for
        log_progress(f"🔎 Searching with cleaned address: '{clean_address}'")
        if clean_address != address:
            log_progress(f"🔄 Original was: '{address}'")
        
        address_search_results_raw = google_search_entity(address_search_query)
        
        # Variant address search - use cleaned address
        address_search_query_variant = f"address {clean_address}"
        address_search_results_raw_variant = google_search_entity(address_search_query_variant)

        # Check if searches failed
        if address_search_results_raw is None:
            address_search_results_raw = "No address search results available"
        
        if address_search_results_raw_variant is None:
            address_search_results_raw_variant = "No variant address search results available"
        
        # Log success if at least one search succeeded
        if (address_search_results_raw != "No address search results available" or 
            address_search_results_raw_variant != "No variant address search results available"):
            log_progress(f"✅ Address search completed")
        else:
            log_progress(f"❌ Both address searches failed")
            
    except Exception as search_error:
        log_progress(f"❌ Address search error: {str(search_error)}")
        address_search_results_raw = f"Address search failed: {str(search_error)}"
        address_search_results_raw_variant = f"Variant address search failed: {str(search_error)}"
    
    # Step 2: LLM Analysis for occupant identification
    log_progress(f"🤖 Analyzing occupant...")
    
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

    occupant_chain = prompt | llm

    try:
        verified_occupant_response = occupant_chain.invoke({}).content.strip()
        log_progress(f"✅ Occupant analysis completed")

        # Parse the response using regex
        confirmed_occupant_match = re.search(r"Selected Occupant:\s*(.*?)\s*---End of Confirmed Occupant---", 
                                           verified_occupant_response, re.DOTALL)
        confirmed_occupant = confirmed_occupant_match.group(1).strip() if confirmed_occupant_match else "Need more information"

        verification_analysis_match = re.search(r"Verification Analysis:\s*(.*?)\s*---End of Verification---", 
                                              verified_occupant_response, re.DOTALL)
        verification_analysis = verification_analysis_match.group(1).strip() if verification_analysis_match else "Analysis not available"
        
    except Exception as llm_error:
        log_progress(f"❌ Analysis failed: {str(llm_error)}")
        confirmed_occupant = "Error"
        verification_analysis = f"LLM analysis failed: {str(llm_error)}"

    # Step 3: Compliance assessment if occupant is identified
    if confirmed_occupant == "Need more information":
        compliance_level = "Need more information"
        rationale = "Unable to confirm occupant, compliance assessment not performed."
        confirmed_occupant_google_search_results = "No occupant identified for search"
    else:
        log_progress(f"🔍 Searching for occupant information...")

        # Google Search for Occupant
        try:
            confirmed_occupant_google_search_results = google_search_entity(confirmed_occupant)
            
            if confirmed_occupant_google_search_results is None:
                confirmed_occupant_google_search_results = "No occupant search results available"
                log_progress(f"⚠️ Occupant search failed")
            else:
                log_progress(f"✅ Occupant search completed")
                    
        except Exception as search_error:
            confirmed_occupant_google_search_results = f"Occupant search failed: {str(search_error)}"
            log_progress(f"⚠️ Occupant search failed")

        # Compliance assessment
        log_progress(f"⚖️ Assessing compliance...")
        
        compliance_prompt = f"""
Assess the occupant's operations based on the following information:

### Selected Occupant: {confirmed_occupant}

Google Search Result of Occupant's name: {confirmed_occupant_google_search_results}
Verification Analysis: {verification_analysis}

Primary approved use: {primary_approved_use}
Secondary approved use: {secondary_approved_use}

Evaluate whether the occupant’s business operations are reasonably aligned with the approved use classification based on standard land use interpretations in Singapore.


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
            log_progress(f"✅ Compliance assessment completed")

            # Parse compliance response using regex
            compliance_match = re.search(r"(?i)\**Compliance Level:\s*(.*?)\s*-{3,}End of Compliance-{3,}", 
                                       verified_compliance_response, re.DOTALL)
            rationale_match = re.search(r"(?i)\**Rationale:\s*(.*?)\s*-{3,}End of Rationale-{3,}", 
                                      verified_compliance_response, re.DOTALL)

            compliance_level = compliance_match.group(1).strip() if compliance_match else "Assessment failed"
            rationale = rationale_match.group(1).strip() if rationale_match else "Rationale not available"
            
        except Exception as compliance_error:
            log_progress(f"❌ Compliance assessment failed")
            compliance_level = "Assessment failed"
            rationale = f"Compliance assessment failed: {str(compliance_error)}"
    
    log_progress(f"✅ Processing completed for {address}")
    
    return {
        'address': address,
        'confirmed_occupant': confirmed_occupant,
        'verification_analysis': verification_analysis,
        'compliance_level': compliance_level,
        'rationale': rationale,
        'google_address_search_results': address_search_results_raw if address_search_results_raw else "N/A",
        'google_address_search_results_variant': address_search_results_raw_variant if address_search_results_raw_variant else "N/A",
        'confirmed_occupant_google_search_results': confirmed_occupant_google_search_results if confirmed_occupant_google_search_results else "N/A"
    }


def process_addresses_batch(addresses, llm, primary_approved_use_list=None, secondary_approved_use_list=None, progress_callback=None):
    """
    Process multiple addresses in batch.
    
    Args:
        addresses: List of addresses to process
        llm: Language model instance
        primary_approved_use_list: List of primary approved uses (same length as addresses)
        secondary_approved_use_list: List of secondary approved uses (same length as addresses)
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Tuple of (results_list, csv_buffer)
    """
    try:
        results = []
        total_addresses = len(addresses)
        
        # Handle default values for approved use lists
        if primary_approved_use_list is None:
            primary_approved_use_list = [""] * len(addresses)
        if secondary_approved_use_list is None:
            secondary_approved_use_list = [""] * len(addresses)
        
        def batch_progress_callback(message):
            if progress_callback:
                progress_callback(message, current_index=len(results) + 1, total=total_addresses)
        
        for i, address in enumerate(addresses):
            try:
                # Get the corresponding approved uses for this address
                primary_use = primary_approved_use_list[i] if i < len(primary_approved_use_list) else ""
                secondary_use = secondary_approved_use_list[i] if i < len(secondary_approved_use_list) else ""
                
                result = process_single_address(address, llm, primary_use, secondary_use, batch_progress_callback)
                results.append([
                    result['address'],
                    result['confirmed_occupant'],
                    result['verification_analysis'],
                    result['compliance_level'],
                    result['rationale'],
                    result['google_address_search_results'],
                    result['google_address_search_results_variant'],
                    result['confirmed_occupant_google_search_results']
                ])
            except Exception as e:
                error_msg = f"❌ Error processing {address}: {str(e)}"
                if progress_callback:
                    progress_callback(error_msg, current_index=i+1, total=total_addresses)
                
                # Add error result to maintain structure
                results.append([
                    address,
                    "Error",
                    str(e),
                    "Unknown",
                    f"Processing failed: {str(e)}",
                    "N/A",
                    "N/A",
                    "N/A"
                ])
    
        if progress_callback:
            progress_callback(f"🎉 Completed processing {len(results)} address(es)!")
        
        # Create CSV buffer
        csv_buffer = create_csv_for_download(results)
        
        # Debug: Ensure we're not returning None
        print(f"🔍 Debug: About to return results={type(results)}, csv_buffer={type(csv_buffer)}")
        
        return results, csv_buffer
        
    except Exception as e:
        # If any error occurs in the entire batch processing, return error results
        print(f"❌ Critical error in batch processing: {str(e)}")
        if progress_callback:
            progress_callback(f"❌ Critical error: {str(e)}")
        
        # Return empty results with error message
        error_results = [[
            "Error",
            "Critical processing failure",
            str(e),
            "Unknown",
            f"Batch processing failed: {str(e)}",
            "N/A",
            "N/A",
            "N/A"
        ]]
        
        error_csv_buffer = create_csv_for_download(error_results)
        
        # Debug: Ensure we're not returning None in error case
        print(f"🔍 Debug Error: About to return error_results={type(error_results)}, error_csv_buffer={type(error_csv_buffer)}")
        
        return error_results, error_csv_buffer
