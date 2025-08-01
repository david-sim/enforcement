"""
Enhanced CSV processor for generating downloadable results with comprehensive data columns.
Includes website extraction, scraping, and advanced compliance assessment with structured JSON outputs.
"""
import pandas as pd
import streamlit as st
import datetime
import re
import io
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pydantic import BaseModel
from typing import Optional

# Define structured output schemas
class OccupantResult(BaseModel):
    """Schema for occupant identification results"""
    confirmed_occupant: str
    matched_snippet: str
    reasoning: str

class ComplianceResult(BaseModel):
    """Schema for compliance assessment results"""
    compliance_level: str
    rationale: str

class BusinessSummary(BaseModel):
    """Schema for business activity summary"""
    business_summary: str
    relevant_keywords: str

def extract_company_website(google_results, known_address):
    """
    Extract company website URL from Google search results, excluding business directories.
    
    Args:
        google_results (str): Google search results text
        known_address (str): Address to verify against website content
        
    Returns:
        str or None: Verified company website URL or None if not found
    """
    urls = re.findall(r'https?://[^\s>"\'\)\]]+', google_results)

    # Domains to exclude (business directories)
    excluded_domains = [
        "yelp.com", "opengovsg.com", "yellowpages.com.sg", "sgpbusiness.com",
        "manufacturingtomorrow.com", "recordowl.com", "companies.sg", "streetdirectory.com",
        "sg.ltddir.com", "www.keepital.com", "singaporebullionmarket.com", "tellme.sg"
    ]

    verified_websites = []

    for url in urls:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Skip excluded domains
            if any(excluded in domain for excluded in excluded_domains):
                continue

            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = soup.get_text(separator=' ', strip=True).lower()
            
            if known_address.lower() in page_text:
                # Address match found, return root domain
                root_url = f"{parsed.scheme}://{parsed.netloc}/"
                verified_websites.append(root_url)
                
        except Exception as e:
            continue  # Skip if request fails

    return verified_websites[0] if verified_websites else None

def scrape_website(url):
    """
    Scrape website content for business activity analysis.
    
    Args:
        url (str): Website URL to scrape
        
    Returns:
        str: Scraped text content (truncated to 3000 chars) or empty string if failed
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad status
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=' ', strip=True)
        return text[:3000]  # avoid overload
    except requests.exceptions.RequestException as req_err:
        print(f"Request error: {req_err}")
    except Exception as e:
        print(f"General scraping error: {e}")
    return ""

def summarize_business_activity(text, llm):
    """
    Summarize business activities from scraped website content using structured output.
    
    Args:
        text (str): Scraped website content
        llm: Language model instance
        
    Returns:
        str: Business activity summary
    """
    # Get compliance rules for context
    try:
        compliance_rules = st.secrets.get("INDUSTRIAL_COMPLIANCE_RULES", "")
    except:
        compliance_rules = ""
    
    summary_prompt = f"""
Below is some extracted website content. Summarize the core business activities and suggest relevant land use classification keywords.

<content>
{text}
</content>

Provide your summary in the following JSON structure:

{{
    "business_summary": "Concise business description focusing on core activities",
    "relevant_keywords": "List of nouns/verbs that indicate core functions and potential B1 use alignment"
}}
"""
    
    # Create structured output LLM for business summary
    structured_summary_llm = llm.with_structured_output(BusinessSummary)
    
    try:
        # Create the full prompt with compliance rules context
        full_summary_prompt = f"{compliance_rules}\n\n{summary_prompt}"
        
        # Get structured summary response
        summary_result = structured_summary_llm.invoke(full_summary_prompt)
        
        # Format the structured result as a readable summary
        formatted_summary = f"Business Summary: {summary_result.business_summary}\nRelevant Keywords: {summary_result.relevant_keywords}"
        return formatted_summary
    except Exception as e:
        return f"Business summary failed: {str(e)}"

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
    Incorporates Step 2 flow: website extraction, scraping, and advanced compliance assessment.
    
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
    
    for i, address in enumerate(addresses, 1):
        try:
            log_progress(f"[{i}/{total_addresses}] 📍 Processing {address}")
            
            # Step 1: Google search for address
            log_progress(f"[{i}/{total_addresses}] 🔍 Searching for address information...")
            
            address_search_results = ""
            try:
                from simple_google_search import simple_google_search
                address_search_results = simple_google_search(f"address {address}", max_retries=1, timeout_seconds=45)
                
                if address_search_results is None:
                    log_progress(f"[{i}/{total_addresses}] ⚠️ Address search failed - trying fallback", show_in_ui=False)
                    from google_search import google_search_entity
                    address_search_results = google_search_entity(f"address {address}")
                    
                if address_search_results is None:
                    address_search_results = "No address search results available"
                    log_progress(f"[{i}/{total_addresses}] ❌ Address search failed")
                else:
                    log_progress(f"[{i}/{total_addresses}] ✅ Address search completed")
                        
            except Exception as search_error:
                log_progress(f"[{i}/{total_addresses}] ❌ Address search error: {str(search_error)}", show_in_ui=False)
                address_search_results = f"Address search failed: {str(search_error)}"
                log_progress(f"[{i}/{total_addresses}] ❌ Address search failed")
            
            # Step 2: LLM Analysis for occupant identification with structured output
            log_progress(f"[{i}/{total_addresses}] 🤖 Analyzing occupant...")
            
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
            
            today = datetime.date.today().isoformat()
            
            occupant_prompt = f"""
Today's date is {today}
Identify the current occupant of: {address} using the search results below.

<google_search_results_original>
{address_search_results}
</google_search_results_original>

Follow the step-by-step instructions in the system prompt and provide your analysis in the following JSON structure:

{{
    "confirmed_occupant": "Business name from snippets or 'Need more information'",
    "matched_snippet": "Quote the relevant snippets that match the address, including URL and source credibility assessment",
    "reasoning": "Show your responses for each step. Why that entity was chosen, or why no match could be confirmed"
}}
"""

            # Create structured output LLM
            structured_llm = llm.with_structured_output(OccupantResult)
            
            try:
                # Get occupant rules for system prompt
                system_rules = get_occupant_rules()
                
                # Create the full prompt
                full_prompt = f"{system_rules}\n\n{occupant_prompt}"
                
                # Get structured response
                occupant_result = structured_llm.invoke(full_prompt)
                log_progress(f"[{i}/{total_addresses}] ✅ Occupant analysis completed")
                
                # Extract structured data
                confirmed_occupant = occupant_result.confirmed_occupant
                matched_snippet = occupant_result.matched_snippet
                reasoning = occupant_result.reasoning
                verification_analysis = f"- Matched snippet(s): {matched_snippet}\n- Reasoning: {reasoning}"
                
                # Debug: Log structured results
                log_progress(f"[{i}/{total_addresses}] 🔍 Structured Occupant: '{confirmed_occupant}'", show_in_ui=False)
                log_progress(f"[{i}/{total_addresses}] 🔍 Structured Analysis: '{verification_analysis[:100]}...'", show_in_ui=False)
                
            except Exception as llm_error:
                log_progress(f"[{i}/{total_addresses}] ❌ LLM analysis failed: {str(llm_error)}", show_in_ui=False)
                log_progress(f"[{i}/{total_addresses}] ❌ Analysis failed")
                confirmed_occupant = "Error"
                matched_snippet = f"LLM analysis failed: {str(llm_error)}"
                reasoning = "Processing error occurred"
                verification_analysis = f"- Matched snippet(s): {matched_snippet}\n- Reasoning: {reasoning}"
            
            # Initialize compliance variables
            compliance_level = "Need more information"
            rationale = "Unable to confirm occupant, compliance assessment not performed."
            occupant_search_results = "No occupant identified for search"
            
            # Step 2: Enhanced compliance assessment if occupant is identified
            if confirmed_occupant and confirmed_occupant != "Unable to determine" and confirmed_occupant != "Need more information":
                log_progress(f"[{i}/{total_addresses}] 🔍 Searching for occupant information...")
                
                # Step 2a: Google search for occupant
                try:
                    from simple_google_search import simple_google_search
                    occupant_search_results = simple_google_search(confirmed_occupant, max_retries=1, timeout_seconds=30)
                    
                    if occupant_search_results is None:
                        log_progress(f"[{i}/{total_addresses}] ⚠️ Occupant search failed - trying fallback", show_in_ui=False)
                        from google_search import google_search_entity
                        occupant_search_results = google_search_entity(confirmed_occupant)
                        
                    if occupant_search_results is None:
                        occupant_search_results = "No occupant search results available"
                        log_progress(f"[{i}/{total_addresses}] ⚠️ Occupant search failed")
                    else:
                        log_progress(f"[{i}/{total_addresses}] ✅ Occupant search completed")
                            
                except Exception as search_error:
                    log_progress(f"[{i}/{total_addresses}] ❌ Occupant search error: {str(search_error)}", show_in_ui=False)
                    occupant_search_results = f"Occupant search failed: {str(search_error)}"
                    log_progress(f"[{i}/{total_addresses}] ⚠️ Occupant search failed")
                
                # Step 2b: Extract company website URL from matched snippets
                log_progress(f"[{i}/{total_addresses}] 🌐 Extracting company website...")
                company_url = extract_company_website(matched_snippet, address)
                
                # Step 2c: Scrape website or use search results for compliance assessment
                use_scraped = False
                scraped_content = ""
                business_summary = ""
                
                if company_url:
                    log_progress(f"[{i}/{total_addresses}] 📄 Scraping website content...")
                    scraped_content = scrape_website(company_url)
                    use_scraped = bool(scraped_content.strip())
                    
                    if use_scraped:
                        log_progress(f"[{i}/{total_addresses}] 📊 Analyzing business activities...")
                        business_summary = summarize_business_activity(scraped_content, llm)
                        log_progress(f"[{i}/{total_addresses}] ✅ Website analysis completed")
                    else:
                        log_progress(f"[{i}/{total_addresses}] ⚠️ Website scraping failed, using search results")
                else:
                    log_progress(f"[{i}/{total_addresses}] ⚠️ No company website found, using search results")
                
                # Step 2d: Compliance assessment with enhanced prompt
                log_progress(f"[{i}/{total_addresses}] ⚖️ Assessing compliance...")
                
                if use_scraped and business_summary:
                    compliance_prompt = f"""
Assess the occupant's operations based on the following information:

### Selected Occupant: {confirmed_occupant}

Scraped Website Content of Occupant: {business_summary}

Then evaluate whether the occupant's business operations are reasonably aligned with B1 use based on standard land use interpretations in Singapore.

Provide your assessment in the following JSON structure:

{{
    "compliance_level": "One of: Unauthorised Use, Authorised Use, Likely Authorised Use, Likely Unauthorised Use, Need more information",
    "rationale": "Detailed rationale for compliance level with specific references to B1 use categories"
}}
"""
                else:
                    compliance_prompt = f"""
Assess the occupant's operations based on the following information:

### Selected Occupant: {confirmed_occupant}

Google Search Result of Occupant: {occupant_search_results}

Then evaluate whether the occupant's business operations are reasonably aligned with B1 use based on standard land use interpretations in Singapore.

Provide your assessment in the following JSON structure:

{{
    "compliance_level": "One of: Unauthorised Use, Authorised Use, Likely Authorised Use, Likely Unauthorised Use, Need more information",
    "rationale": "Detailed rationale for compliance level with specific references to B1 use categories"
}}
"""

                # Create structured output LLM for compliance
                structured_compliance_llm = llm.with_structured_output(ComplianceResult)

                try:
                    # Get compliance rules for system context
                    compliance_rules = get_compliance_rules()
                    
                    # Create the full prompt
                    full_compliance_prompt = f"{compliance_rules}\n\n{compliance_prompt}"
                    
                    # Get structured compliance response
                    compliance_result = structured_compliance_llm.invoke(full_compliance_prompt)
                    log_progress(f"[{i}/{total_addresses}] ✅ Compliance assessment completed")
                    
                    # Extract structured compliance data
                    compliance_level = compliance_result.compliance_level
                    rationale = compliance_result.rationale
                    
                    # Debug: Log structured compliance results
                    log_progress(f"[{i}/{total_addresses}] 🔍 Structured Compliance: '{compliance_level}'", show_in_ui=False)
                    log_progress(f"[{i}/{total_addresses}] 🔍 Structured Rationale: '{rationale[:100]}...'", show_in_ui=False)
                    
                except Exception as compliance_error:
                    log_progress(f"[{i}/{total_addresses}] ❌ Compliance assessment failed: {str(compliance_error)}", show_in_ui=False)
                    log_progress(f"[{i}/{total_addresses}] ❌ Compliance assessment failed")
                    compliance_level = "Assessment failed"
                    rationale = f"Compliance assessment failed: {str(compliance_error)}"
            
            # Create result row with all 7 columns
            result = [
                address,                              # address
                confirmed_occupant,                   # confirmed_occupant
                verification_analysis,                # verification_analysis
                compliance_level,                     # compliance_level
                rationale,                           # rationale
                address_search_results,              # google_address_search_results
                occupant_search_results              # confirmed_occupant_google_search_results
            ]
            
            results.append(result)
            log_progress(f"[{i}/{total_addresses}] ✅ Processing completed")
            
        except Exception as e:
            error_msg = f"[{i}/{total_addresses}] ❌ Error processing {address}: {str(e)}"
            log_progress(error_msg, show_in_ui=False)
            log_progress(f"[{i}/{total_addresses}] ❌ Processing failed")
            
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
