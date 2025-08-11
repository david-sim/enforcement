# ⚖️ Enforcement Processing Tool

An AI-powered enforcement processing system that analyzes addresses for compliance checking and occupant identification. The tool provides comprehensive automated analysis with both bulk processing capabilities and interactive single-record entry options.

## 🚀 Key Features

### 📊 Dual Processing Modes
- **CSV Bulk Processing**: Upload CSV files containing multiple addresses for batch analysis
- **Single Record Entry**: Manually enter individual addresses for quick analysis
- **Smart Input Validation**: Real-time form validation and user guidance

### 🤖 AI-Powered Analysis
- **Occupant Identification**: Automated detection of current occupants using advanced AI
- **Compliance Assessment**: Comprehensive evaluation against approved use regulations  
- **Google Search Integration**: Real-time web searches for address verification
- **Enforcement Priority Scoring**: Intelligent prioritization of enforcement actions

### 💬 Interactive Chat Interface
- **Conversational AI**: Engage with the enforcement assistant for guidance and support
- **Context-Aware Responses**: AI maintains conversation history for coherent interactions
- **Real-Time Assistance**: Get help with processing questions and technical issues

### 📋 Comprehensive Reporting
- **Detailed Results**: Complete analysis with rationale and recommendations
- **CSV Export**: Download results in structured CSV format
- **PDF Summary Reports**: Generate comprehensive summary reports
- **Visual Analytics**: Charts and statistics for processed data
- **Persistent Results**: Results remain visible for multiple downloads until inputs change

### 🔧 Advanced Features
- **Session State Management**: Smart handling of user data and results persistence
- **Progress Tracking**: Real-time processing updates with detailed logs
- **Error Handling**: Robust error management with user-friendly feedback
- **Multi-Page Navigation**: Clean interface with About and Methodology sections

## 🏗️ System Requirements

- **Python**: 3.10 or higher
- **Dependencies**: All required packages listed in `requirements.txt`
- **API Access**: OpenAI API key required for AI processing

## ⚙️ Setup Instructions

### 1. API Configuration
Configure your OpenAI API key using one of these methods:
- **Streamlit Secrets**: Add to `secrets.toml`
  ```toml
  OPENAI_API_KEY = "your-api-key-here"
  ```
- **Environment Variable**: 
  ```bash
  export OPENAI_API_KEY="your-api-key-here"
  ```

### 2. Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Application
```bash
streamlit run streamlit_main.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage Guide

### Bulk CSV Processing
1. Select "Upload CSV file" option
2. Upload your CSV file with address data
3. Choose address type (shophouse/industrial)  
4. Click "🚀 Start Processing CSV File"
5. Monitor real-time progress and download results

**Expected CSV Format:**
```
Column 1: Address (Required) - Complete address with postal code
Column 2: Primary Approved Use (Optional) - e.g., "Shophouse", "Industrial"
Column 3: Secondary Approved Use (Optional) - e.g., "Retail", "Manufacturing"
```

### Single Record Entry
1. Select "Enter single record manually" option
2. Fill in the form fields:
   - **Address** (Required): Complete address including unit and postal code
   - **Primary Approved Use** (Optional): Primary use designation
   - **Secondary Approved Use** (Optional): Secondary use designation
3. Click "Validate Entry" to confirm input
4. Select address type (shophouse/industrial)
5. Click "🚀 Process Single Record"

### Interactive Chat
- Navigate to the chat interface in the sidebar
- Ask questions about processing, results, or general assistance
- The AI maintains conversation context for better interactions

## 📊 Output Features

### Results Display
- **Persistent Results**: Results remain visible after downloads
- **Smart Clearing**: Results only clear when inputs change
- **Multiple Downloads**: Download CSV and PDF reports multiple times
- **Visual Analytics**: Charts showing compliance statistics and trends

### Report Types
- **CSV Results**: Structured data with all analysis details
- **TXT Summary**: Comprehensive report with statistics and insights
- **Real-time Logs**: Processing progress and detailed execution logs

## 🏛️ Architecture

The application follows a modular architecture:
- `streamlit_main.py`: Main application entry point
- `ui_components.py`: UI utilities and Streamlit components  
- `enforcement_engine.py`: Core processing logic
- `csv_processor.py`: CSV handling and validation
- `search_service.py`: Google search integration
- Page modules: `about_page.py`, `methodology_page.py`

## 🔒 License

Released under the [MIT License](LICENSE). See the `LICENSE` file for details.
