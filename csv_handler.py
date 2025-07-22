import pandas as pd
import streamlit as st
import datetime
import re
from google_search import google_search_entity
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_occupant_rules():
    """Get occupant rules from secrets, with fallback for testing"""
    try:
        return st.secrets.get("OCCUPANT_RULES", "")
    except:
        return ""

today = datetime.date.today().isoformat()

def process_industrial_addresses(addresses, llm, chat_callback=None):
    """
    Process a list of industrial addresses, search for occupants, and return results.
    """
    output_data = []
    for i, address in enumerate(addresses):
        address_search_query = f"address {address}"
        address_search_results_raw = google_search_entity(address_search_query, chat_callback=chat_callback)

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
- Reasoning: (Show your responses for each step. Why that entity was chosen, or why no match could be confirmed)
---End of Verification---
"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(get_occupant_rules()),
            HumanMessagePromptTemplate.from_template(occupant_prompt)
        ])
        occupant_chain = prompt | llm

        if chat_callback:
            chat_callback(f"🤖 Asking LLM to identify occupant for: {address}")

        verified_occupant_response = occupant_chain.invoke({}).content.strip()

        # Display LLM response in chat
        if chat_callback:
            chat_callback(f"LLM Response for {address}:\n{verified_occupant_response}")

        confirmed_occupant_match = re.search(
            r"Selected Occupant:\s*(.*?)\s*---End of Confirmed Occupant---",
            verified_occupant_response, re.DOTALL)
        confirmed_occupant = confirmed_occupant_match.group(1).strip() if confirmed_occupant_match else None

        verification_analysis_match = re.search(
            r"Verification Analysis:\s*(.*?)\s*---End of Verification---",
            verified_occupant_response, re.DOTALL)
        verification_analysis = verification_analysis_match.group(1).strip() if verification_analysis_match else None

        output_data.append([
            address, confirmed_occupant, verification_analysis,
            address_search_results_raw if address_search_results_raw else "N/A"
        ])
    return output_data

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