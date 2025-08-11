"""
Core enforcement processing engine.
Contains the main address processing logic without UI dependencies.
"""
import datetime
import re
import pandas as pd
import io
import json
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from pydantic import BaseModel, Field
from search_service import google_search_entity
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from csv_processor import process_csv_with_validation, create_csv_for_download, CSVValidationError


class OccupantResult(BaseModel):
    """Structured result for occupant identification analysis."""
    confirmed_occupant: str = Field(description="The confirmed occupant name or 'Need more information'")
    matched_snippet: str = Field(description="Relevant snippets that match the address with sources and credibility")
    business_summary: str = Field(description="Summary of the core and other business activities of the selected occupant")
    reasoning: str = Field(description="Detailed reasoning for the occupant selection decision")


class ComplianceResult(BaseModel):
    """Structured result for compliance assessment analysis."""
    compliance_level: str = Field(description="The compliance level: Unauthorised Use / Authorised Use / Likely Authorised Use / Likely Unauthorised Use / Need more information")
    rationale: str = Field(description="Detailed rationale explaining the compliance level determination")


def parse_json_response(response: str) -> Optional[dict]:
    """
    Parse JSON response from LLM with robust error handling.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    try:
        # Clean the response - remove leading/trailing whitespace
        response = response.strip()
        
        # Handle potential code block markers
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        # Parse JSON
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing JSON: {e}")
        return None


def generate_variant(address):
    """
    Generate a variant address format for shophouse addresses.
    Converts between unit notation (#01-XX) and suffix notation (A, B, C, D).
    
    Args:
        address: The original address string
        
    Returns:
        Variant address string or None if no conversion is possible
    """
    unit_pattern = re.match(r"^(\d+)\s+(.*)\s+#(0[1-5])-\d{2}\s+(Singapore\s+\d{6})$", address)
    if unit_pattern:
        block = unit_pattern.group(1)
        street = unit_pattern.group(2).strip()
        floor = unit_pattern.group(3)
        postal = unit_pattern.group(4)

        floor_to_suffix = {"01" : "", "02": "A", "03": "B", "04": "C", "05": "D"}
        suffix = floor_to_suffix.get(floor)
        if suffix:
            return f"{block}{suffix} {street} {postal}"
        else:
            return None

    suffix_pattern = re.match(r"^(\d+)([A-D])\s+(.*)\s+(Singapore\s+\d{6})$", address)
    if suffix_pattern:
        block = suffix_pattern.group(1)
        suffix = suffix_pattern.group(2)
        street = suffix_pattern.group(3).strip()
        postal = suffix_pattern.group(4)

        suffix_to_floor = {"A": "02", "B": "03", "C": "04", "D": "05"}
        floor = suffix_to_floor.get(suffix)
        if floor:
            return f"{block} {street} #{floor}-01 {postal}"
        else:
            return None

    return None


def get_occupant_rules(address_type: str = "industrial") -> str:
    """Get occupant rules from secrets based on address type, with fallback for testing"""
    try:
        import streamlit as st
        if address_type.lower() == "shophouse":
            return st.secrets.get("SHOPHOUSE_OCCUPANT_RULES", "")
        else:
            return st.secrets.get("INDUSTRIAL_OCCUPANT_RULES", "")
    except:
        return ""


def get_compliance_rules(address_type: str = "industrial") -> str:
    """Get compliance rules from secrets based on address type, with fallback for testing"""
    try:
        import streamlit as st
        if address_type.lower() == "shophouse":
            return st.secrets.get("SHOPHOUSE_COMPLIANCE_RULES", "")
        else:
            return st.secrets.get("INDUSTRIAL_COMPLIANCE_RULES", "")
    except:
        return ""


def process_csv(command: str, csv_file) -> Dict[str, List[str]]:
    """
    Process the uploaded CSV file based on the command ("shophouse" or "industrial").
    This function now uses the modular csv_processor for validation and processing.

    Args:
        command: The command, either "shophouse" or "industrial"
        csv_file: The uploaded CSV file object

    Returns:
        Dictionary containing the processed data
        
    Raises:
        CSVValidationError: If CSV validation fails
        ValueError: If command is invalid
    """
    try:
        return process_csv_with_validation(command, csv_file)
    except CSVValidationError as e:
        # Convert CSV validation errors to ValueError for backward compatibility
        raise ValueError(f"CSV validation failed: {str(e)}")


def process_single_address(address: str, llm: Any, primary_approved_use: str = "", secondary_approved_use: str = "", address_type: str = "industrial", progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    """
    Process a single address through the complete enforcement workflow.
    
    Args:
        address: Address to process
        llm: Language model instance
        primary_approved_use: Primary approved use for the address
        secondary_approved_use: Secondary approved use for the address
        address_type: Type of address processing ("shophouse" or "industrial")
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
    
    # Get system rules based on address type
    occupant_rules = get_occupant_rules(address_type)
    compliance_rules = get_compliance_rules(address_type)
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
        
        # Variant address search - different logic for shophouse vs industrial
        if address_type.lower() == "shophouse":
            # For shophouse, try to generate an address format variant
            variant_address = generate_variant(clean_address)
            if variant_address:
                address_search_query_variant = variant_address
                log_progress(f"🔄 Generated shophouse variant: '{variant_address}'")
            else:
                # Fallback to standard variant if no format conversion possible
                address_search_query_variant = f"{clean_address}"
                log_progress(f"🔄 Using fallback variant for shophouse: 'address {clean_address}'")
        else:
            # For industrial, use the standard variant
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
    
    # Define JSON structure outside f-string to avoid template variable conflicts
    occupant_json_structure = """{{
    "confirmed_occupant": "Business name from snippets or 'Need more information'",
    "matched_snippet": "Quote the relevant snippets that match the address, including URL and source credibility assessment",
    "business_summary": "Summarize the core and other business activities of the selected occupant",
    "reasoning": "Show your responses for each step. Why that entity was chosen, or why no match could be confirmed"
}}"""
    
    occupant_prompt = f"""Today's date is {today}
Identify the current occupant of: {address} using the search results below.

<google_search_results_original>
{address_search_results_raw}
</google_search_results_original>

<google_search_results_variant>
{address_search_results_raw_variant}
</google_search_results_variant>

---

Follow the step-by-step instructions in the system prompt and provide your analysis in the following JSON structure:
{occupant_json_structure}
"""

    # Create ChatPromptTemplate with system rules and human prompt
    occupant_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", occupant_rules),
        ("human", occupant_prompt)
    ])

    # Create structured LLM chain
    structured_llm = llm.with_structured_output(OccupantResult)
    occupant_chain = occupant_chat_prompt | structured_llm

    try:
        occupant_result = occupant_chain.invoke({})
        log_progress(f"✅ Occupant analysis completed")
        
        confirmed_occupant = occupant_result.confirmed_occupant
        verification_analysis = f"Matched Snippets: {occupant_result.matched_snippet}\n\nBusiness Summary: {occupant_result.business_summary}\n\nReasoning: {occupant_result.reasoning}"
        
    except Exception as llm_error:
        log_progress(f"❌ Analysis failed: {str(llm_error)}")
        confirmed_occupant = "Analysis not available"
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
        
        # Define JSON structure outside f-string to avoid template variable conflicts
        compliance_json_structure = """{{
    "compliance_level": "One of: Unauthorised Use, Authorised Use, Likely Authorised Use, Likely Unauthorised Use, Need more information",
    "rationale": "Detailed rationale for compliance level with specific references to B1 use categories"
}}"""
        
        compliance_prompt = f"""Assess the occupant's operations based on the following information:

### Selected Occupant: {confirmed_occupant}

Google Search Result of Occupant's name: {confirmed_occupant_google_search_results}
Verification Analysis: {verification_analysis}

Evaluate whether the occupant's business operations are reasonably aligned with the approved use classification based on standard land use interpretations in Singapore.

Primary approved use: {primary_approved_use}
Secondary approved use: {secondary_approved_use}

---

Provide your assessment in the following JSON structure:
{compliance_json_structure}
"""

        # Create ChatPromptTemplate with system rules and human prompt
        compliance_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", compliance_rules),
            ("human", compliance_prompt)
        ])

        # Create structured LLM chain
        structured_compliance_llm = llm.with_structured_output(ComplianceResult)
        compliance_chain = compliance_chat_prompt | structured_compliance_llm

        try:
            compliance_result = compliance_chain.invoke({})
            log_progress(f"✅ Compliance assessment completed")
            
            compliance_level = compliance_result.compliance_level
            rationale = compliance_result.rationale
            
        except Exception as compliance_error:
            log_progress(f"❌ Compliance assessment failed")
            compliance_level = "Assessment failed"
            rationale = f"Compliance assessment failed: {str(compliance_error)}"
    
    log_progress(f"✅ Processing completed for {address}")
    
    return {
        'address': address,
        'confirmed_occupant': confirmed_occupant,
        'verification_analysis': verification_analysis,
        'primary_approved_use': primary_approved_use,
        'secondary_approved_use': secondary_approved_use,
        'compliance_level': compliance_level,
        'rationale': rationale,
        'google_address_search_results': address_search_results_raw if address_search_results_raw else "N/A",
        'google_address_search_results_variant': address_search_results_raw_variant if address_search_results_raw_variant else "N/A",
        'confirmed_occupant_google_search_results': confirmed_occupant_google_search_results if confirmed_occupant_google_search_results else "N/A"
    }


def process_addresses_batch(addresses, llm, primary_approved_use_list=None, secondary_approved_use_list=None, address_type="industrial", progress_callback=None):
    """
    Process multiple addresses in batch.
    
    Args:
        addresses: List of addresses to process
        llm: Language model instance
        primary_approved_use_list: List of primary approved uses (same length as addresses)
        secondary_approved_use_list: List of secondary approved uses (same length as addresses)
        address_type: Type of address processing ("shophouse" or "industrial")
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
                
                result = process_single_address(address, llm, primary_use, secondary_use, address_type, batch_progress_callback)
                results.append([
                    result['address'],
                    result['confirmed_occupant'],
                    result['verification_analysis'],
                    result['primary_approved_use'],
                    result['secondary_approved_use'],
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
                primary_use = primary_approved_use_list[i] if i < len(primary_approved_use_list) else ""
                secondary_use = secondary_approved_use_list[i] if i < len(secondary_approved_use_list) else ""
                
                results.append([
                    address,
                    "Error",
                    str(e),
                    primary_use,
                    secondary_use,
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
            "",  # primary_approved_use
            "",  # secondary_approved_use
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
