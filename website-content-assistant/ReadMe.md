# Website Content Assistant

A powerful AI-driven tool that transforms any website into a conversational interface. Simply enter a website URL and get instant, intelligent answers about the content through natural language queries.

## 🚀 Features

- **Universal Web Scraping**: Extract content from any publicly accessible website
- **Intelligent Q&A**: Ask natural language questions about the scraped content
- **Vector-Based Search**: Uses advanced embeddings for semantic content understanding
- **Real-time Processing**: Instant content analysis and question answering
- **Clean Interface**: Simple, user-friendly Streamlit interface
- **Conversation Memory**: Maintains context across multiple questions

## 🛠️ Technology Stack

- **LangChain**: Framework for building LLM applications
- **Streamlit**: Web application framework
- **OpenAI GPT**: Language model for intelligent responses
- **FAISS**: Vector database for efficient similarity search
- **Beautiful Soup**: Web scraping and HTML parsing
- **Sentence Transformers**: Text embeddings generation

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd website-content-assistant
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

## 🔧 Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.28.0
langchain>=0.0.350
langchain-openai>=0.0.5
beautifulsoup4>=4.12.0
requests>=2.31.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
python-dotenv>=1.0.0
lxml>=4.9.0
html2text>=2020.1.16
```

## 🚀 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter any website URL in the input field

4. Wait for the content to be processed and indexed

5. Start asking questions about the website content!

## 💡 Example Use Cases

- **Research**: "What are the main services offered by this company?"
- **Product Information**: "What are the key features of their flagship product?"
- **Contact Details**: "How can I get in touch with their support team?"
- **Content Analysis**: "What is the main topic discussed in this article?"
- **Comparison**: "What are the differences between their pricing plans?"

## 🏗️ Architecture

1. **Web Scraping**: Uses Beautiful Soup to extract clean text content from websites
2. **Text Processing**: Chunks content into manageable pieces for better processing
3. **Embedding Generation**: Creates vector embeddings using sentence transformers
4. **Vector Storage**: Stores embeddings in FAISS for efficient similarity search
5. **Question Answering**: Uses LangChain with OpenAI GPT to generate contextual answers

## 📋 Features Roadmap

- [ ] Support for authentication-protected websites
- [ ] Bulk URL processing
- [ ] Export conversation history
- [ ] Custom embedding models
- [ ] Multi-language support
- [ ] Website content caching
- [ ] Advanced filtering options

## 🔒 Security & Privacy

- Only processes publicly accessible web content
- No data is permanently stored
- API keys are securely managed through environment variables
- Respects robots.txt and website scraping policies

## 🛡️ Limitations

- Cannot access password-protected or authentication-required content
- Some websites may block automated scraping
- Processing time depends on website size and complexity
- Rate limiting may apply for large websites

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support, please contact [your-email@example.com]

## 🏷️ Tags

`langchain` `streamlit` `ai` `nlp` `web-scraping` `question-answering` `rag` `openai` `vector-search`

---

**Built with LangChain** - Showcasing the power of AI-driven content understanding and conversational interfaces.