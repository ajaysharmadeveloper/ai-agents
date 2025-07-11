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

## 📁 Project Structure
```
document-intelligence-poc/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── docs/
│   └── README.md          # Documentation
└── tests/
    └── test_app.py        # Unit tests
```

## 🔧 Configuration Options

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### Advanced LangChain Configuration
```python
# Custom embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Custom LLM settings
llm = OpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo-instruct",
    max_tokens=2000
)
```

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add secrets in Streamlit Cloud:
   - `OPENAI_API_KEY = "your_api_key"`
5. Deploy with one click

### Option 2: AWS EC2 Deployment
```bash
# Install on EC2 instance
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt

# Run with nohup for background execution
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
```

### Option 3: Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t doc-intelligence-poc .
docker run -p 8501:8501 doc-intelligence-poc
```

## 🧪 Testing the POC

### Sample Test Cases
1. **PDF Upload**: Test with technical documentation
2. **DOCX Upload**: Test with business documents
3. **TXT Upload**: Test with plain text files
4. **Question Variety**: Ask different types of questions:
   - Factual: "What is the main topic?"
   - Analytical: "What are the key recommendations?"
   - Specific: "What dates are mentioned?"
   - Summarization: "Can you summarize this document?"

### Performance Metrics to Track
- Document processing time
- Question response time
- Token usage per query
- Cost per interaction
- User engagement metrics

## 📊 Demo Script for Clients

### Opening Hook
"Let me show you how AI can instantly understand any document you upload. This isn't just document storage - it's document intelligence."

### Demo Flow
1. **Upload a Complex Document** (e.g., policy manual, technical guide)
2. **Ask Strategic Questions**:
   - "What are the main compliance requirements?"
   - "Who should I contact for exceptions?"
   - "What are the deadlines mentioned?"
3. **Show Source Attribution**: Highlight how it shows exactly where answers come from
4. **Demonstrate Conversation Memory**: Ask follow-up questions
5. **Show Multi-format Support**: Upload different file types

### Key Benefits to Highlight
- **Time Savings**: "Instead of spending hours reading manuals, get instant answers"
- **Accuracy**: "AI reads every word and finds relevant information"
- **Accessibility**: "Make any document searchable and conversational"
- **Scalability**: "Handle hundreds of documents with the same ease"

## 🎯 Customization Options

### Industry-Specific Modifications
1. **Legal**: Add legal document parsing, clause extraction
2. **Healthcare**: Add medical terminology processing
3. **Finance**: Add financial document analysis, compliance checking
4. **HR**: Add policy Q&A, employee handbook assistance

### Technical Enhancements
1. **Multiple Document Support**: Upload and query multiple documents
2. **Document Comparison**: Compare content across documents
3. **Export Features**: Export Q&A sessions to PDF
4. **Integration APIs**: Connect to existing document management systems

## 🔐 Security Considerations

### Data Privacy
- Documents are processed in memory only
- No permanent storage of uploaded content
- API keys are handled securely
- Consider adding user authentication for production

### Production Hardening
```python
# Add input validation
def validate_file_size(file):
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        raise ValueError("File too large")

# Add rate limiting
import time
from functools import wraps

def rate_limit(max_calls=10, time_window=60):
    def decorator(func):
        calls = []
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [call for call in calls if call > now - time_window]
            if len(calls) >= max_calls:
                raise Exception("Rate limit exceeded")
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

## 📈 Success Metrics

### Technical KPIs
- Average document processing time < 30 seconds
- Query response time < 5 seconds
- 95% accuracy in answer relevance
- Support for documents up to 10MB

### Business KPIs
- Lead generation from demo usage
- Client engagement time > 5 minutes
- Conversion rate from demo to consultation
- Customer satisfaction score > 4.5/5

## 🎨 Branding & Customization

### Add Your Company Branding
```python
# Custom CSS for branding
st.markdown("""
<style>
.main-header {
    color: #your-brand-color;
}
.upload-section {
    background: linear-gradient(135deg, #your-color1, #your-color2);
}
</style>
""", unsafe_allow_html=True)
```

### Custom Footer
```python
st.markdown("""
---
<div style="text-align: center;">
    <p>Built with ❤️ by [Your Company Name]</p>
    <p>🚀 Want to build something similar? <a href="mailto:your@email.com">Get in touch!</a></p>
</div>
""", unsafe_allow_html=True)
```

## 📞 Lead Capture Integration

### Add Contact Form
```python
with st.expander("🤝 Interested in Custom AI Solutions?"):
    st.subheader("Let's discuss your AI needs")
    
    name = st.text_input("Name")
    email = st.text_input("Email")
    company = st.text_input("Company")
    message = st.text_area("Tell us about your project")
    
    if st.button("Send Message"):
        # Integration with CRM/email service
        send_lead_notification(name, email, company, message)
        st.success("Thanks! We'll be in touch soon.")
```

This POC is designed to be:
- **Quick to deploy** (under 1 hour)
- **Impressive to demo** (immediate wow factor)
- **Easy to customize** (modular code structure)
- **Lead generating** (built-in contact capture)
- **Scalable** (foundation for more complex projects)

Ready to build this out? Let me know if you'd like to modify any aspects or add specific features!