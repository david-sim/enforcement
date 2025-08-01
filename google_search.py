from serpapi import GoogleSearch
import streamlit as st
import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def get_serpapi_key():
    """Get SERPAPI key from secrets, with fallback for testing"""
    try:
        return st.secrets.get("SERPAPI_API_KEY", None)
    except:
        return None

def perform_search_with_timeout(search_params, timeout=30):
    """Perform Google search with timeout handling."""
    def _search():
        try:
            search = GoogleSearch(search_params)
            return search.get_dict()
        except Exception as e:
            return {"error": f"Search API error: {str(e)}"}
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_search)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                return {"error": f"Search request timed out after {timeout} seconds"}
    except Exception as e:
        return {"error": f"Search executor error: {str(e)}"}

# Google Search results
def google_search_entity(address, chat_callback=None, progress_handler=None):
    print(f"Executing Google Search for: {address}")
    
    # Validate API key first
    api_key = get_serpapi_key()
    if not api_key:
        error_msg = "SERPAPI_API_KEY not found in secrets"
        print(error_msg)
        if progress_handler:
            progress_handler.update_progress(f"❌ {error_msg}", increment=False)
        return None

    if chat_callback:
        chat_callback(f"🔎 Searching Google for: {address}")
    
    if progress_handler:
        progress_handler.update_progress(f"🔎 Initiating Google search for: {address}", increment=False)
    
    try:
        search_params = {
            "q": f"address {address}",  # Add "address" prefix for better results
            "location": "Singapore",
            "hl": "en",
            "gl": "sg",
            "filter": "0",
            "api_key": api_key
        }
        
        if progress_handler:
            progress_handler.update_progress(f"📡 Sending API request to Google...", increment=False)
        
        # Perform search with timeout
        results = perform_search_with_timeout(search_params, timeout=45)
        
        if "error" in results:
            error_msg = f"Error in SERP response: {results['error']}"
            print(error_msg)
            logging.error(error_msg)
            if progress_handler:
                progress_handler.update_progress(f"❌ {error_msg}", increment=False)
            return None

        organic_results = results.get("organic_results", [])
        
        if progress_handler:
            progress_handler.update_progress(f"✅ Found {len(organic_results)} search results", increment=False)

        if chat_callback:
            chat_callback(f"🔎 Complete Google for: {address}")

        formatted_results = format_structured_results(parse_results(organic_results))
        
        # Return empty string if no results found, rather than None
        return formatted_results if formatted_results.strip() else "No search results found for this address"
        
    except Exception as e:
        error_msg = f"Unexpected error during Google search: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        if progress_handler:
            progress_handler.update_progress(f"❌ {error_msg}", increment=False)
        return None

def parse_results(raw_results):
    return raw_results if isinstance(raw_results, list) else []

def format_structured_results(parsed_results):
    return "\n".join([
        f"Title: {item.get('title', '')}\nLink: {item.get('link', '')}\nSnippet: {item.get('snippet', '')}\nDate: {item.get('date', 'No date available')}\n"
        for item in parsed_results
    ])