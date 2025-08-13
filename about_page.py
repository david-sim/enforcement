"""
About Us page for the Smart Compliance Operations Unit Tool.
Contains project information, scope, objectives, and methodology.
"""
import streamlit as st


def display_about_page():
    """Display the About Us page content."""
    
    # Page header
    st.markdown("# 🔍 About Us")
    st.markdown("---")
    
    # Project Overview Section
    st.markdown("## 📋 Project Overview")
    
    st.markdown("""
    **Smart Compliance Operations Unit Tool** is an AI-assisted prototype developed to revolutionise 
    the way unit addresses are processed for occupant identification and compliance detection 
    of unauthorised use.
    
    This innovative solution combines advanced artificial intelligence with automated 
    processing capabilities to streamline smart compliance operations and ensure regulatory compliance.
    """)
    
    # Project Scope Section
    st.markdown("## 🎯 Project Scope")
    
    st.markdown("""
    The scope of this project encompasses the development of an intelligent system that:
    
    • **Resolves Unit Addresses**: Automatically processes and standardises address formats
    
    • **Identifies Occupants**: Uses AI to determine current occupants of specified addresses
    
    • **Detects Compliance Issues**: Identifies potential unauthorised use of spaces
    
    • **Automates Assessment**: Reduces manual processing time and human error
    
    • **Provides Actionable Insights**: Generates comprehensive reports for enforcement actions
    """)
    
    # Objective Section
    st.markdown("## 🎖️ Objective")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ### Primary Goals:
        1. **Occupant Identification**
        2. **Compliance Assessment**
        3. **Automated Processing**
        """)
    
    with col2:
        st.markdown("""
        The primary objective is to create a two-stage automated process:
        
        **Stage 1: Occupant Determination**  
        Accurately identify the current occupant of each unit address using AI-powered 
        analysis of available data sources.
        
        **Stage 2: Compliance Verification**  
        Check whether the identified occupant's use of the space aligns with the 
        approved use designation for that address.
        """)
    
    # LLM Implementation Section
    st.markdown("## 🤖 LLM Model Implementation")
    
    st.markdown("""
    The Large Language Model (LLM) is strategically implemented across **three critical areas** 
    to ensure comprehensive and accurate processing:
    """)
    
    # Create three columns for the LLM areas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🕵️ Area 1: Occupant Determination
        
        **Purpose**: Identify current occupants
        
        **Process**:
        - Analyse search results and data sources
        - Cross-reference multiple information points
        - Apply intelligent matching algorithms
        - Verify occupant authenticity
        
        **Output**: Confirmed occupant name or "Need more information"
        """)
    
    with col2:
        st.markdown("""
        ### 🏢 Area 2: Business Activity Assessment
        
        **Purpose**: Understand nature of business operations
        
        **Process**:
        - Research occupant's business activities
        - Analyse operational characteristics
        - Categorise business type and functions
        - Assess scale and scope of operations
        
        **Output**: Detailed business activity profile
        """)
    
    with col3:
        st.markdown("""
        ### ⚖️ Area 3: Compliance Verification
        
        **Purpose**: Check alignment with approved use
        
        **Process**:
        - Compare business activities with approved use
        - Apply regulatory compliance rules
        - Assess compatibility and violations
        - Generate compliance determination
        
        **Output**: Compliance level and detailed rationale
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666; font-style: italic;'>
    Smart Compliance Operations Unit Tool - AI-Assisted Compliance Management System
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    display_about_page()
