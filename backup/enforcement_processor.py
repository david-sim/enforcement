"""
Legacy wrapper for enforcement processing functionality.
Maintains backward compatibility while delegating to the new modular engine.
"""
import pandas as pd
import streamlit as st
import datetime
import re
import io
from enforcement_engine import (
    process_csv,
    create_csv_for_download,
    process_addresses_batch,
    get_occupant_rules,
    get_compliance_rules
)


def process_industrial_addresses_enhanced(addresses, llm):
    """
    Legacy wrapper for enhanced processing of industrial addresses.
    Maintains compatibility with existing code while using the new engine.
    
    Args:
        addresses: List of addresses to process
        llm: Language model instance
    
    Returns:
        Tuple of (results, progress_messages, csv_buffer)
    """
    progress_messages = []
    
    def progress_callback(message, current_index=None, total=None):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        if current_index and total:
            formatted_message = f"🔄 [{timestamp}] [{current_index}/{total}] {message}"
        else:
            formatted_message = f"🔄 [{timestamp}] {message}"
        
        print(formatted_message)
        progress_messages.append(f"🔄 {message}")
    
    # Use the new modular engine
    results, csv_buffer = process_addresses_batch(addresses, llm, progress_callback)
    
    return results, progress_messages, csv_buffer
