"""
Enhanced CSV processor for generating downloadable results with comprehensive data columns.
Simplified flow: address search -> occupant identification -> compliance assessment using verification analysis.
"""
import pandas as pd
import streamlit as st
import datetime
import re
import io

def get_occupant_rules():
    """Get occupant rules from secrets, with fallback for testing"""
    try:
        return st.secrets.get("INDUSTRIAL_OCCUPANT_RULES", "")
    except:
        return ""

def get_compliance_rules():
    """Get compliance rules from secrets, with fallback for testing"""
    try:
        return st.secrets.get("INDUSTRIAL_COMPLIANCE_RULES", "")
    except:
        return ""

def process_csv(command: str, csv_file) -> dict:
    """
    Process the uploaded CSV file based on the command ("shophouse" or "industrial").

    Args:
        command (str): The command, either "shophouse" or "industrial".
        csv_file: The uploaded CSV file object.

    Returns:
        dict: A dictionary containing the processed data.
            - For "shophouse": {"addresses": [...], "primary_approved_use": [...], "secondary_approved_use": [...]}
            - For "industrial": {"addresses": [...]}
    """
    csv_file.seek(0)
    df = pd.read_csv(csv_file, header=0)  # Skip header row

    if command.lower() == "shophouse":
        addresses = df.iloc[:, 0].dropna().tolist()
        primary_approved_use = df.iloc[:, 1].dropna().tolist() if df.shape[1] > 1 else []
        secondary_approved_use = df.iloc[:, 2].dropna().tolist() if df.shape[1] > 2 else []
        return {
            "addresses": addresses,
            "primary_approved_use": primary_approved_use,
            "secondary_approved_use": secondary_approved_use
        }
    elif command.lower() == "industrial":
        addresses = df.iloc[:, 0].dropna().tolist()
        return {
            "addresses": addresses
        }
    else:
        raise ValueError("Invalid command. Must be 'shophouse' or 'industrial'.")

def create_csv_for_download(results_data):
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
        'confirmed_occupant_google_search_results'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(results_data, columns=columns)
    
    # Create CSV buffer
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)
    
    return csv_buffer

def process_industrial_addresses_enhanced(addresses, llm):
    """
    Enhanced processing of industrial addresses with comprehensive data collection.
    Simplified flow: address search -> occupant identification -> compliance assessment using verification analysis.
    
    Args:
        addresses: List of addresses to process
        llm: Language model instance
    
    Returns:
        Tuple of (results, progress_messages, csv_buffer)
    """
    results = []
    progress_messages = []
    total_addresses = len(addresses)
    
    def log_progress(message, show_in_ui=True):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"🔄 [{timestamp}] {message}"
        
        # Always log to console
        print(formatted_message)
        
        # Only add important messages to UI
        if show_in_ui:
            ui_message = f"🔄 {message}"
            progress_messages.append(ui_message)
    
    log_progress(f"🚀 Starting enhanced processing of {total_addresses} address(es)...")
    
    # Get system rules
    occupant_rules = get_occupant_rules()
    compliance_rules = get_compliance_rules()
    today = datetime.date.today().isoformat()
    
    for i in range(len(addresses)):
        try:
            address = addresses[i]
            log_progress(f"[{i+1}/{total_addresses}] 📍 Processing {address}")
            
            # Initialize variables
            address_search_results_raw = ""
            confirmed_occupant_google_search_results = ""
            verified_occupant_response = ""
            verification_analysis = ""
            confirmed_occupant = ""
            compliance_level = "Need more information"
            rationale = "Unable to confirm occupant, compliance assessment not performed."
            
            # Step 1: Google search for address
            log_progress(f"[{i+1}/{total_addresses}] 🔍 Searching for address information...")
            
            try:
                from google_search import google_search_entity
                address_search_query = f"address {address}"
                address_search_results_raw = google_search_entity(address_search_query)
                
                if address_search_results_raw is None:
                    address_search_results_raw = "No address search results available"
                    log_progress(f"[{i+1}/{total_addresses}] ❌ Address search failed")
                else:
                    log_progress(f"[{i+1}/{total_addresses}] ✅ Address search completed")
                        
            except Exception as search_error:
                log_progress(f"[{i+1}/{total_addresses}] ❌ Address search error: {str(search_error)}", show_in_ui=False)
                address_search_results_raw = f"Address search failed: {str(search_error)}"
                log_progress(f"[{i+1}/{total_addresses}] ❌ Address search failed")
            
            # Step 2: LLM Analysis for occupant identification
            log_progress(f"[{i+1}/{total_addresses}] 🤖 Analyzing occupant...")
            
            from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
            
            occupant_prompt = f"""
Today's date is {today}
Identify the current occupant of: {address} using the search results below.

<google_search_results_original>
{address_search_results_raw}
</google_search_results_original>

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
                log_progress(f"[{i+1}/{total_addresses}] ✅ Occupant analysis completed")
                
                print(f"Verified Occupant Response: {verified_occupant_response}")

                # Parse the response using regex
                confirmed_occupant_match = re.search(r"Selected Occupant:\s*(.*?)\s*---End of Confirmed Occupant---", verified_occupant_response, re.DOTALL)
                confirmed_occupant = confirmed_occupant_match.group(1).strip() if confirmed_occupant_match else "Need more information"

                verification_analysis_match = re.search(r"Verification Analysis:\s*(.*?)\s*---End of Verification---", verified_occupant_response, re.DOTALL)
                verification_analysis = verification_analysis_match.group(1).strip() if verification_analysis_match else "Analysis not available"
                
                # Debug: Log parsed results
                log_progress(f"[{i+1}/{total_addresses}] 🔍 Confirmed Occupant: '{confirmed_occupant}'", show_in_ui=False)
                log_progress(f"[{i+1}/{total_addresses}] 🔍 Verification Analysis: '{verification_analysis[:100]}...'", show_in_ui=False)
                
            except Exception as llm_error:
                log_progress(f"[{i+1}/{total_addresses}] ❌ LLM analysis failed: {str(llm_error)}", show_in_ui=False)
                log_progress(f"[{i+1}/{total_addresses}] ❌ Analysis failed")
                confirmed_occupant = "Error"
                verification_analysis = f"LLM analysis failed: {str(llm_error)}"

            # Step 3: Compliance assessment if occupant is identified
            if confirmed_occupant == "Need more information":
                compliance_level = "Need more information"
                rationale = "Unable to confirm occupant, compliance assessment not performed."
                confirmed_occupant_google_search_results = "No occupant identified for search"
            else:
                log_progress(f"[{i+1}/{total_addresses}] 🔍 Searching for occupant information...")
                print(f"Selected Occupant: {confirmed_occupant}")

                # Step 2: Google Search for Occupant
                try:
                    from google_search import google_search_entity
                    confirmed_occupant_google_search_results = google_search_entity(confirmed_occupant)
                    
                    if confirmed_occupant_google_search_results is None:
                        confirmed_occupant_google_search_results = "No occupant search results available"
                        log_progress(f"[{i+1}/{total_addresses}] ⚠️ Occupant search failed")
                    else:
                        log_progress(f"[{i+1}/{total_addresses}] ✅ Occupant search completed")
                            
                except Exception as search_error:
                    log_progress(f"[{i+1}/{total_addresses}] ❌ Occupant search error: {str(search_error)}", show_in_ui=False)
                    confirmed_occupant_google_search_results = f"Occupant search failed: {str(search_error)}"
                    log_progress(f"[{i+1}/{total_addresses}] ⚠️ Occupant search failed")

                # Step 3: Compliance assessment
                log_progress(f"[{i+1}/{total_addresses}] ⚖️ Assessing compliance...")
                
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
                    log_progress(f"[{i+1}/{total_addresses}] ✅ Compliance assessment completed")
                    
                    print(f"\nGenerated Compliance Assessment:\n{verified_compliance_response}")

                    # Parse compliance response using regex
                    compliance_match = re.search(r"(?i)\**Compliance Level:\s*(.*?)\s*-{3,}End of Compliance-{3,}", verified_compliance_response, re.DOTALL)
                    rationale_match = re.search(r"(?i)\**Rationale:\s*(.*?)\s*-{3,}End of Rationale-{3,}", verified_compliance_response, re.DOTALL)

                    compliance_level = compliance_match.group(1).strip() if compliance_match else "Assessment failed"
                    rationale = rationale_match.group(1).strip() if rationale_match else "Rationale not available"
                    
                    # Debug: Log compliance results
                    log_progress(f"[{i+1}/{total_addresses}] 🔍 Compliance Level: '{compliance_level}'", show_in_ui=False)
                    log_progress(f"[{i+1}/{total_addresses}] 🔍 Rationale: '{rationale[:100]}...'", show_in_ui=False)
                    
                except Exception as compliance_error:
                    log_progress(f"[{i+1}/{total_addresses}] ❌ Compliance assessment failed: {str(compliance_error)}", show_in_ui=False)
                    log_progress(f"[{i+1}/{total_addresses}] ❌ Compliance assessment failed")
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
            log_progress(f"[{i+1}/{total_addresses}] ✅ Processing completed")
            
        except Exception as e:
            error_msg = f"[{i+1}/{total_addresses}] ❌ Error processing {address}: {str(e)}"
            log_progress(error_msg, show_in_ui=False)
            log_progress(f"[{i+1}/{total_addresses}] ❌ Processing failed")
            
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
    
    log_progress(f"✅ Completed processing {len(results)} address(es)!")
    
    # Create CSV buffer
    csv_buffer = create_csv_for_download(results)
    
    return results, progress_messages, csv_buffer
