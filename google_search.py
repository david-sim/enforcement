from serpapi import GoogleSearch
import streamlit as st
import time

def get_serpapi_key():
    """Get SERPAPI key from secrets, with fallback for testing"""
    try:
        return st.secrets.get("SERPAPI_API_KEY", None)
    except:
        return None

# Google Search results
def google_search_entity(address, chat_callback=None):
    print(f"Executing Google Search for: {address}")

    if chat_callback:
        chat_callback(f"🔎 Searching Google for: {address}")
    
    time.sleep(3)
    search = GoogleSearch({
        "q": f"{address}",
        "location": "Singapore",
        "hl": "en",
        "gl": "sg",
        "filter": "0",
        "api_key": get_serpapi_key()
    })
    results = search.get_dict()

    if "error" in results:
        print(f"Error in SERP response: {results['error']}")
        return None

    organic_results = results.get("organic_results", [])

    if chat_callback:
        chat_callback(f"🔎 Complete Google for: {address}")

    return format_structured_results(parse_results(organic_results))

def parse_results(raw_results):
    return raw_results if isinstance(raw_results, list) else []

def format_structured_results(parsed_results):
    return "\n".join([
        f"Title: {item.get('title', '')}\nLink: {item.get('link', '')}\nSnippet: {item.get('snippet', '')}\nDate: {item.get('date', 'No date available')}\n"
        for item in parsed_results
    ])