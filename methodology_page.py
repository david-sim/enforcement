"""
Methodology page for the Smart Compliance Operations Unit Tool.
Contains detailed methodology, technology stack, and benefits information.
"""
import streamlit as st
from PIL import Image


def display_methodology_page():
    """Display the Methodology page content."""
    
    # Page header
    st.markdown("# 🔬 Methodology")
    st.markdown("---")
    
    # Methodology Section
    st.markdown("## 🔬 Processing Methodology")
    
    with st.expander("📊 Processing Workflow", expanded=True):
        st.markdown("""
        ### Step-by-Step Process:
        
        1. **Data Input**: Upload CSV files containing addresses and approved use information
        
        2. **Address Resolution**: Clean and standardise address formats for optimal processing
        
        3. **Information Gathering**: 
           - Perform targeted web searches for address information
           - Collect occupant-related data from multiple sources
           - Generate search result variants for comprehensive coverage
        
        4. **AI Analysis Phase 1 - Occupant Identification**:
           - Process search results using advanced LLM analysis
           - Apply occupant identification rules and criteria
           - Cross-validate findings across data sources
        
        5. **AI Analysis Phase 2 - Business Assessment**:
           - Research identified occupant's business operations
           - Analyse business activities and operational scope
           - Categorise business type and functions
        
        6. **AI Analysis Phase 3 - Compliance Check**:
           - Compare business activities with approved use classifications
           - Apply compliance rules specific to address type (shophouse/industrial)
           - Generate compliance determination with detailed rationale
        
        7. **Results Generation**: 
           - Compile comprehensive processing results
           - Generate downloadable reports and summaries
           - Provide actionable compliance insights
        """)
        
        # Add workflow flowchart illustration
        st.markdown("### 📊 Process Flow Illustration")
        try:
            flowchart_image = Image.open("imgs/process_flowchart_white.png")
            st.image(flowchart_image, caption="Smart Compliance Operations Workflow", width=600)
        except:
            st.warning("Process flowchart image not found")
    
    # Technology Stack Section
    st.markdown("## 💻 Technology Stack")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        ### Core Technologies:
        - **Python**: Primary programming language
        - **Streamlit**: Web application framework
        - **Pandas**: Data processing and analysis
        - **LangChain**: LLM integration and prompt management
        """)
    
    with tech_col2:
        st.markdown("""
        ### AI & Processing:
        - **Large Language Model**: GPT-powered analysis
        - **Search Integration**: Automated web search capabilities
        - **CSV Processing**: Automated file handling and validation
        - **Real-time Processing**: Live progress tracking and updates
        """)
    
    # Benefits Section
    st.markdown("## ✨ Key Benefits")
    
    benefits_col1, benefits_col2, benefits_col3 = st.columns(3)
    
    with benefits_col1:
        st.markdown("""
        ### 🚀 Efficiency
        - Automated processing reduces manual effort
        - Batch processing capabilities
        - Real-time progress tracking
        - Instant results generation
        """)
    
    with benefits_col2:
        st.markdown("""
        ### 🎯 Accuracy
        - AI-powered analysis reduces human error
        - Consistent application of compliance rules
        - Multiple data source verification
        - Comprehensive result validation
        """)
    
    with benefits_col3:
        st.markdown("""
        ### 📊 Insights
        - Detailed compliance assessments
        - Comprehensive summary reports
        - Actionable enforcement recommendations
        - Statistical analysis and trends
        """)
    
    # Implementation Details
    st.markdown("## ⚙️ Implementation Details")
    
    impl_col1, impl_col2 = st.columns(2)
    
    with impl_col1:
        st.markdown("""
        ### Data Processing:
        - **CSV Validation**: Automatic format checking and error reporting
        - **Address Cleaning**: Standardisation of address formats
        - **Batch Processing**: Efficient handling of multiple addresses
        - **Progress Tracking**: Real-time status updates
        """)
    
    with impl_col2:
        st.markdown("""
        ### AI Integration:
        - **Prompt Engineering**: Optimised LLM prompts for accuracy
        - **Chain Processing**: Sequential AI analysis stages
        - **Error Handling**: Robust failure recovery mechanisms
        - **Result Validation**: Quality assurance checks
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666; font-style: italic;'>
    Methodology Documentation - Technical Implementation Guide
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    display_methodology_page()
