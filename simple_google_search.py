"""
Simple and robust Google search with basic timeout handling.
"""
from serpapi import GoogleSearch
import streamlit as st
import time
import logging

def get_serpapi_key():
    """Get SERPAPI key from secrets, with fallback for testing"""
    try:
        return st.secrets.get("SERPAPI_API_KEY", None)
    except:
        return None

def simple_google_search(address, max_retries=2, timeout_seconds=30):
    """
    Simple Google search with basic error handling and retries.
    
    Args:
        address: Address to search for
        max_retries: Maximum number of retry attempts
        timeout_seconds: Not implemented in this simple version, but kept for compatibility
    
    Returns:
        Search results as formatted string or None if failed
    """
    api_key = get_serpapi_key()
    if not api_key:
        print("❌ SERPAPI_API_KEY not found in secrets")
        return None
    
    for attempt in range(max_retries + 1):
        try:
            print(f"🔎 Attempt {attempt + 1}/{max_retries + 1} - Searching Google for: {address}")
            
            search_params = {
                "q": f"address {address}",
                "location": "Singapore",
                "hl": "en",
                "gl": "sg",
                "filter": "0",
                "api_key": api_key
            }
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            if "error" in results:
                error_msg = results["error"]
                print(f"❌ SERP API error: {error_msg}")
                if attempt < max_retries:
                    print(f"⏳ Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    return None
            
            organic_results = results.get("organic_results", [])
            print(f"✅ Found {len(organic_results)} search results")
            
            if not organic_results:
                return "No search results found for this address"
            
            # Format results
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
            
        except Exception as e:
            error_msg = f"Search attempt {attempt + 1} failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            if attempt < max_retries:
                print(f"⏳ Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"❌ All search attempts failed for: {address}")
                return None
    
    return None
