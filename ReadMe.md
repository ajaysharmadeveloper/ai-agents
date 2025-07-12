# Document Intelligence Playground - Setup & Deployment Guide

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Create project directory
mkdir document-intelligence-poc
cd document-intelligence-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 3. Run the Application
```bash
streamlit run app.py
```