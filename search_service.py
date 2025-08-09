"""
Unified Google search service with timeout handling and retry logic.
Combines functionality from google_search.py and simple_google_search.py.
"""
import time
import logging
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Import handling for different environments
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    GoogleSearch = None
    print("⚠️ WARNING: serpapi not installed. Google search functionality will be limited.")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def get_serpapi_key() -> Optional[str]:
    """Get SERPAPI key from secrets, with fallback for testing"""
    if STREAMLIT_AVAILABLE:
        try:
            return st.secrets.get("SERPAPI_API_KEY", None)
        except:
            return None
    
    # Fallback for non-Streamlit environments
    import os
    return os.getenv("SERPAPI_API_KEY")


def perform_search_with_timeout(search_params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """Perform Google search with timeout handling."""
    if not SERPAPI_AVAILABLE or GoogleSearch is None:
        return {"error": "SERPAPI not available. Please install: pip install google-search-results"}
    
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


def google_search_entity(query: str, location: str = "Singapore", max_retries: int = 2, timeout: int = 45) -> Optional[str]:
    """
    Unified Google search with timeout and retry logic.
    
    Args:
        query: Search query string
        location: Location for search context
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each search attempt
    
    Returns:
        Formatted search results as string or None if failed
    """
    print(f"🔎 Executing Google Search for: {query}")
    
    # Validate API key first
    api_key = get_serpapi_key()
    if not api_key:
        error_msg = "SERPAPI_API_KEY not found in secrets"
        print(f"❌ {error_msg}")
        return None

    for attempt in range(max_retries + 1):
        try:
            print(f"🔎 Attempt {attempt + 1}/{max_retries + 1} - Searching for: {query}")
            
            search_params = {
                "q": query,
                "location": location,
                "hl": "en",
                "gl": "sg",
                "filter": "0",
                "api_key": api_key
            }
            
            # Perform search with timeout
            results = perform_search_with_timeout(search_params, timeout=timeout)
            
            if "error" in results:
                error_msg = f"SERP API error: {results['error']}"
                print(f"❌ {error_msg}")
                if attempt < max_retries:
                    print(f"⏳ Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    return None
            
            organic_results = results.get("organic_results", [])
            print(f"✅ Found {len(organic_results)} search results")
            
            if not organic_results:
                return "No search results found for this query"
            
            # Format results using helper function
            return format_search_results(organic_results)
            
        except Exception as e:
            error_msg = f"Search attempt {attempt + 1} failed: {str(e)}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)
            
            if attempt < max_retries:
                print(f"⏳ Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"❌ All search attempts failed for: {query}")
                return None
    
    return None


def format_search_results(organic_results: List[Dict[str, Any]]) -> str:
    """
    Format search results into a consistent string format.
    
    Args:
        organic_results: List of organic search results from SERP API
    
    Returns:
        Formatted search results as string
    """
    if not organic_results:
        return "No search results found"
    
    formatted_results = []
    for item in organic_results:
        title = item.get('title', 'No title')
        link = item.get('link', 'No link')
        snippet = item.get('snippet', 'No snippet')
        date = item.get('date', 'No date available')
        
        formatted_results.append(
            f"Title: {title}\nLink: {link}\nSnippet: {snippet}\nDate: {date}\n"
        )
    
    return "\n".join(formatted_results)


# Legacy compatibility functions
def simple_google_search(address: str, max_retries: int = 2, timeout_seconds: int = 30) -> Optional[str]:
    """Legacy compatibility wrapper for simple_google_search."""
    return google_search_entity(f"{address}", max_retries=max_retries, timeout=timeout_seconds)
