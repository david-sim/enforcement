from serpapi import GoogleSearch
import streamlit as st
import time

SERPAPI_API_KEY = st.secrets.get("SERPAPI_API_KEY", None)

# Google Search results
def google_search_entity(address):
    print(f"Executing Google Search for: {address}")
    time.sleep(3)
    search = GoogleSearch({
        "q": f"{address}",
        "location": "Singapore",
        "hl": "en",
        "gl": "sg",
        "filter": "0",
        "api_key": SERPAPI_API_KEY
    })
    results = search.get_dict()

    if "error" in results:
        print(f"Error in SERP response: {results['error']}")
        return None

    organic_results = results.get("organic_results", [])

    return format_structured_results(parse_results(organic_results))

def parse_results(raw_results):
    return raw_results if isinstance(raw_results, list) else []

def format_structured_results(parsed_results):
    return "\n".join([
        f"Title: {item.get('title', '')}\nLink: {item.get('link', '')}\nSnippet: {item.get('snippet', '')}\nDate: {item.get('date', 'No date available')}\n"
        for item in parsed_results
    ])