# Document Intelligence Playground - Setup & Deployment Guide

## 🧠 Overview

An intelligent document analysis system that allows users to upload any document (PDF, DOCX, TXT) and ask questions about its content using AI. Built with LangChain, OpenAI, and Streamlit to demonstrate advanced RAG (Retrieval-Augmented Generation) capabilities.

## ✨ Features

- **Multi-Format Support**: Upload PDF, DOCX, and TXT files
- **Intelligent Text Chunking**: Optimal document segmentation for better retrieval
- **Semantic Search**: Vector embeddings for context-aware question answering
- **Conversation Memory**: Maintains context across multiple questions
- **Source Attribution**: Shows relevant document sections for each answer
- **Usage Tracking**: Monitor token usage and costs
- **Interactive Chat Interface**: Seamless Q&A experience
- **Real-time Processing**: Instant document analysis and response generation

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

## 📦 Requirements

Create a `requirements.txt` file with:
```text
streamlit
langchain
langchain-openai
langchain-community
openai
PyPDF2
python-docx
faiss-cpu
tiktoken
python-dotenv
```

## 🏗️ Project Structure

```
document-intelligence-poc/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .env                     # Environment variables
├── .gitignore              # Git ignore file
└── uploads/                # Temporary file storage (auto-created)
```

## 🔧 Configuration

### OpenAI API Key
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add it to your `.env` file or enter it in the app sidebar
3. Ensure you have sufficient credits for embeddings and chat completions

### Document Processing Settings
- **Chunk Size**: 1000 characters (adjustable in `RecursiveCharacterTextSplitter`)
- **Chunk Overlap**: 200 characters for context preservation
- **Vector Search**: Returns top 5 most relevant chunks
- **Memory Window**: Maintains full conversation history

## 🎯 Usage Examples

### Sample Questions to Try:
- "What is the main topic of this document?"
- "Can you summarize the key points?"
- "What are the important dates mentioned?"
- "Who are the key people or organizations mentioned?"
- "What actions or recommendations are suggested?"
- "What are the financial figures discussed?"

### Supported File Types:
- **PDF Files**: Extracts text from all pages
- **DOCX Files**: Processes Word documents with formatting
- **TXT Files**: Plain text files with UTF-8 encoding

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your `OPENAI_API_KEY` in the app secrets
4. Deploy with one click

### Local Network Access
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

## 🛠️ Technical Architecture

### Core Components
- **LangChain**: Orchestrates the RAG workflow
- **OpenAI GPT-3.5-turbo**: Powers the question-answering
- **OpenAI Embeddings**: Creates semantic vector representations
- **FAISS Vector Store**: Enables fast similarity search
- **Streamlit**: Provides interactive web interface

### Data Flow
1. User uploads document
2. Text extraction based on file type
3. Document chunking with overlap
4. Vector embeddings generation
5. FAISS index creation
6. User asks questions
7. Semantic search retrieval
8. LLM generates contextual answers

## 🎨 Customization

### Adjusting Chunk Parameters
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Increase for longer chunks
    chunk_overlap=300,    # Increase for more context
    separators=["\n\n", "\n", " ", ""]
)
```

### Modifying Retrieval Settings
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}  # Return more relevant chunks
)
```

### Custom AI Behavior
```python
# Update the conversation chain parameters
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True,
    verbose=True
)
```

## 🔍 Troubleshooting

### Common Issues
1. **OpenAI API Error**: Verify your API key and account credits
2. **File Upload Issues**: Check file size limits and format support
3. **Memory Errors**: Large documents may require chunking adjustments
4. **Import Errors**: Ensure all dependencies match requirements.txt versions

### File Processing Errors
- **PDF**: Install latest PyPDF2 version for better compatibility
- **DOCX**: Ensure python-docx is properly installed
- **Large Files**: Consider splitting documents before upload

### Debug Mode
Add this to your app for debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Analytics & Monitoring

The app tracks:
- Document processing statistics
- Token usage and costs per query
- Number of questions asked
- Source document references
- Processing time metrics

## 🔒 Security Considerations

- **API Keys**: Never commit API keys to version control
- **File Upload**: Validate file types and sizes
- **Data Privacy**: Documents are processed in memory, not stored permanently
- **Access Control**: Consider authentication for production deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

**Built by Nirix AI** - Specializing in LangChain & Document Intelligence Solutions

- 📧 Email: ajaysharmabjki96@gmail.com
- 📞 Phone: +91 9414256219
- 🌐 Website: nirixai.com

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Business Applications

### Use Cases
- **Legal Document Analysis**: Extract key information from contracts and agreements
- **Research Paper Processing**: Quickly understand academic papers and reports
- **Policy Document Review**: Navigate complex policy and procedure documents
- **Manual and Guide Analysis**: Interactive help system for technical documentation
- **Financial Report Analysis**: Extract insights from financial statements
- **Compliance Documentation**: Search and query regulatory documents

### ROI Benefits
- **Time Savings**: Instant document understanding vs manual reading
- **Improved Accuracy**: AI-powered information extraction
- **Enhanced Accessibility**: Make documents searchable and interactive
- **Scalability**: Process multiple documents simultaneously
- **Knowledge Retention**: Maintain institutional knowledge in accessible format

### Integration Possibilities
- **CRM Systems**: Integrate with customer documents
- **Knowledge Management**: Corporate document libraries
- **Customer Support**: Automated document-based help systems
- **Research Tools**: Academic and scientific document analysis

---

*This is a demonstration project showcasing AI-powered document intelligence capabilities. Perfect for client presentations and proof-of-concept demonstrations in the enterprise document processing market.*