import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import os
from dotenv import load_dotenv
import time
from typing import List, Dict, Any
# import html2text  # We'll create a simple alternative if not available

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Website Content Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background: #cce7ff;
        color: #004085;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #b8daff;
    }
</style>
""", unsafe_allow_html=True)


class WebsiteContentAssistant:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key

        # Initialize components only if API key is provided
        if self.openai_api_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            self.llm = ChatOpenAI(
                temperature=0.1,
                model_name="gpt-3.5-turbo",
                openai_api_key=self.openai_api_key
            )
        else:
            self.embeddings = None
            self.llm = None

        # Text splitter for chunking content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def is_valid_url(self, url: str) -> bool:
        """Validate if URL is properly formatted"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def extract_text_from_url(self, url: str) -> Dict[str, Any]:
        """Extract clean text content from a given URL"""
        try:
            # Validate URL first
            if not self.is_valid_url(url):
                return {
                    'title': None,
                    'content': None,
                    'url': url,
                    'success': False,
                    'error': "Invalid URL format"
                }

            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }

            # Make request with timeout
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return {
                    'title': None,
                    'content': None,
                    'url': url,
                    'success': False,
                    'error': f"Unsupported content type: {content_type}"
                }

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()

            # Extract title
            title = "No title found"
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find('h1'):
                title = soup.find('h1').get_text().strip()

            # Extract text content (simple alternative to html2text)
            # Remove unwanted elements first
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'meta', 'link']):
                element.decompose()

            # Get text content
            text_content = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)

            # Additional cleanup
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()

            # Validate content length
            if len(text_content.strip()) < 100:
                return {
                    'title': title,
                    'content': None,
                    'url': url,
                    'success': False,
                    'error': "Website content too short or empty"
                }

            return {
                'title': title,
                'content': text_content,
                'url': url,
                'success': True,
                'error': None
            }

        except requests.Timeout:
            return {
                'title': None,
                'content': None,
                'url': url,
                'success': False,
                'error': "Request timeout - website took too long to respond"
            }
        except requests.ConnectionError:
            return {
                'title': None,
                'content': None,
                'url': url,
                'success': False,
                'error': "Connection error - unable to reach website"
            }
        except requests.HTTPError as e:
            return {
                'title': None,
                'content': None,
                'url': url,
                'success': False,
                'error': f"HTTP error {e.response.status_code}: {e.response.reason}"
            }
        except requests.RequestException as e:
            return {
                'title': None,
                'content': None,
                'url': url,
                'success': False,
                'error': f"Request failed: {str(e)}"
            }
        except Exception as e:
            return {
                'title': None,
                'content': None,
                'url': url,
                'success': False,
                'error': f"Parsing failed: {str(e)}"
            }

    def process_content(self, content_data: Dict[str, Any]) -> FAISS:
        """Process extracted content and create vector store"""
        if not self.openai_api_key:
            raise Exception("OpenAI API key is required")

        if not content_data['success']:
            raise Exception(content_data['error'])

        # Create document
        doc = Document(
            page_content=content_data['content'],
            metadata={
                'title': content_data['title'],
                'url': content_data['url'],
                'source': 'web_scraping'
            }
        )

        # Split text into chunks
        chunks = self.text_splitter.split_documents([doc])

        if not chunks:
            raise Exception("No content could be extracted from the website")

        # Create vector store
        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        return vectorstore

    def create_qa_chain(self, vectorstore: FAISS) -> ConversationalRetrievalChain:
        """Create conversational QA chain with memory"""
        if not self.openai_api_key:
            raise Exception("OpenAI API key is required")

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",  # Specify which output key to store in memory
            k=5  # Remember last 5 exchanges
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True,
            verbose=False  # Disable verbose to reduce noise
        )

        return qa_chain


def main():
    # Header (ONLY shown once)
    st.markdown("""
    <div class="main-header">
        <h1>🌐 Website Content Assistant</h1>
        <p>Transform any website into a conversational interface</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None

    # Initialize session state
    if 'processed_url' not in st.session_state:
        st.session_state.processed_url = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'website_info' not in st.session_state:
        st.session_state.website_info = None
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    if 'selected_url' not in st.session_state:
        st.session_state.selected_url = ""

    # Sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")

        # OpenAI API Key input
        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            placeholder="sk-...",
            help="Your API key is not stored and only used for this session",
            key="sidebar_api_key"
        )

        if api_key:
            if api_key.startswith('sk-') and len(api_key) > 20:
                if not st.session_state.api_key_valid or st.session_state.assistant is None:
                    st.session_state.assistant = WebsiteContentAssistant(api_key)
                    st.session_state.api_key_valid = True
                st.success("✅ API Key loaded successfully!")
            else:
                st.error("❌ Invalid API key format")
                st.session_state.api_key_valid = False
        else:
            st.info("Please enter your OpenAI API key to continue")
            st.session_state.api_key_valid = False

        st.markdown("---")

        st.header("📋 How to Use")
        st.markdown("""
        <div class="info-box">
        <strong>Step 1:</strong> Enter your OpenAI API key<br>
        <strong>Step 2:</strong> Enter any website URL<br>
        <strong>Step 3:</strong> Wait for content processing<br>
        <strong>Step 4:</strong> Ask questions about the content<br>
        <strong>Step 5:</strong> Get intelligent answers!
        </div>
        """, unsafe_allow_html=True)

        st.header("✨ Features")
        features = [
            "🔍 Smart content extraction",
            "🤖 AI-powered Q&A",
            "💬 Conversation memory",
            "⚡ Real-time processing",
            "🎯 Contextual answers"
        ]
        for feature in features:
            st.markdown(f"• {feature}")

        st.header("💡 Example Questions")
        examples = [
            "What is this website about?",
            "What services do they offer?",
            "How can I contact them?",
            "What are the main features?",
            "Summarize the key points"
        ]
        for example in examples:
            st.markdown(f"• _{example}_")

    # Main content area
    if not st.session_state.api_key_valid:
        st.header("🔑 Welcome to Website Content Assistant")
        st.markdown("""
        <div class="info-box">
        <h3>Get Started</h3>
        <p>To use this application, please enter your OpenAI API key in the sidebar.</p>
        <p><strong>Don't have an API key?</strong> Get one from <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI Platform</a></p>
        <p><strong>Security:</strong> Your API key is only used for this session and is not stored anywhere.</p>
        </div>
        """, unsafe_allow_html=True)

        # URL input section (always visible)
        st.header("🌐 Enter Website URL")

        col1, col2 = st.columns([3, 1])

        with col1:
            url_input = st.text_input(
                "Website URL to analyze:",
                value=st.session_state.selected_url,
                placeholder="https://example.com",
                help="Enter any publicly accessible website URL",
                key="welcome_url_input"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            analyze_button = st.button("🚀 Analyze Website", type="primary", disabled=True)
            if analyze_button:
                st.toast("⚠️ Please enter your OpenAI API key first!", icon="⚠️")

        # Example URLs section
        st.header("📚 Try These Example Websites")
        st.markdown("Click any example below to auto-fill the URL field:")

        # Create example URLs with descriptions
        example_urls = [
            {
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "title": "🧠 Wikipedia - Artificial Intelligence",
                "description": "Learn about AI concepts and history"
            },
            {
                "url": "https://python.langchain.com/docs/get_started/introduction",
                "title": "🔗 LangChain Documentation",
                "description": "Explore LangChain framework docs"
            },
            {
                "url": "https://streamlit.io/",
                "title": "⚡ Streamlit Homepage",
                "description": "Discover Streamlit app framework"
            },
            {
                "url": "https://openai.com/",
                "title": "🤖 OpenAI Homepage",
                "description": "Learn about OpenAI and their AI models"
            },
            {
                "url": "https://docs.python.org/3/",
                "title": "🐍 Python Documentation",
                "description": "Official Python 3 documentation"
            },
            {
                "url": "https://github.com/",
                "title": "💻 GitHub Homepage",
                "description": "Explore the world's largest code repository"
            }
        ]

        # Display examples in a nice grid
        cols = st.columns(2)
        for i, example in enumerate(example_urls):
            with cols[i % 2]:
                if st.button(
                        f"{example['title']}",
                        help=example['description'],
                        key=f"example_{i}",
                        use_container_width=True
                ):
                    # Show error toast and update selected_url
                    st.toast("⚠️ Please enter your OpenAI API key first to analyze websites!", icon="⚠️")
                    st.session_state.selected_url = example['url']
                    st.rerun()
                st.caption(example['description'])

        st.header("🌐 What This App Does")
        features = [
            "🔍 **Smart Web Scraping**: Extract content from any public website",
            "🤖 **AI-Powered Q&A**: Ask natural language questions about the content",
            "💬 **Conversation Memory**: Maintains context across multiple questions",
            "⚡ **Real-time Processing**: Instant content analysis and responses",
            "🎯 **Contextual Answers**: Get precise answers based on website content"
        ]

        for feature in features:
            st.markdown(feature)
    else:
        render_main_app_content()

    # Always render footer sections regardless of API key status
    render_footer_sections()


def render_main_app_content():
    """Render the main application content when API key is valid"""

    # Main content columns
    main_col1, main_col2 = st.columns([2, 1])

    with main_col1:
        st.header("🔗 Website URL")

        # Check if URL was set from welcome screen or auto-processing
        default_url = st.session_state.selected_url

        url_input = st.text_input(
            "Enter the website URL you want to analyze:",
            value=default_url,
            placeholder="https://example.com",
            help="Enter any publicly accessible website URL",
            key="main_url_input"
        )

        # Button columns
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            process_button = st.button("🚀 Process Website", type="primary", key="process_btn")
        with btn_col2:
            clear_button = st.button("🗑️ Clear Session", key="clear_btn")

    with main_col2:
        if st.session_state.website_info:
            st.header("📊 Website Info")
            info = st.session_state.website_info
            st.markdown(f"**Title:** {info['title']}")
            st.markdown(f"**URL:** {info['url']}")
            st.markdown(f"**Status:** ✅ Processed")
            st.markdown(f"**Content Length:** {len(info['content'])} characters")

    # Clear session
    if clear_button:
        for key in ['processed_url', 'vectorstore', 'qa_chain', 'chat_history', 'website_info']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.selected_url = ""
        st.rerun()

    # Handle auto-processing from quick examples
    if 'auto_process_url' in st.session_state and st.session_state.auto_process_url:
        url_input = st.session_state.auto_process_url
        # Store the selected URL for the input field
        st.session_state.selected_url = url_input
        del st.session_state.auto_process_url  # Clear the flag

        # Force a rerun to update the input field, then process
        if 'processing_started' not in st.session_state:
            st.session_state.processing_started = True
            st.rerun()

        # Auto-trigger processing
        with st.spinner("🔄 Processing website content..."):
            try:
                # Check if API key is still valid
                if not st.session_state.api_key_valid:
                    st.toast("⚠️ Please enter your OpenAI API key first!", icon="⚠️")
                    if 'processing_started' in st.session_state:
                        del st.session_state.processing_started
                    return

                # Extract content
                content_data = st.session_state.assistant.extract_text_from_url(url_input)

                if not content_data['success']:
                    st.error(f"❌ Failed to process website: {content_data['error']}")
                    # Clear the processing flag
                    if 'processing_started' in st.session_state:
                        del st.session_state.processing_started
                else:
                    # Process content and create vector store
                    vectorstore = st.session_state.assistant.process_content(content_data)

                    # Create QA chain
                    qa_chain = st.session_state.assistant.create_qa_chain(vectorstore)

                    # Store in session state
                    st.session_state.processed_url = url_input
                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_chain = qa_chain
                    st.session_state.website_info = content_data
                    st.session_state.chat_history = []

                    st.success(
                        f"✅ Website '{content_data['title']}' processed successfully! You can now ask questions below.")

                    # Clear the processing flag
                    if 'processing_started' in st.session_state:
                        del st.session_state.processing_started

            except Exception as e:
                st.error(f"❌ Error processing content: {str(e)}")
                st.toast(f"❌ Error: {str(e)}", icon="❌")
                # Clear the processing flag and any partial state
                if 'processing_started' in st.session_state:
                    del st.session_state.processing_started
                for key in ['vectorstore', 'qa_chain', 'website_info']:
                    if key in st.session_state:
                        del st.session_state[key]

    # Process website
    if process_button and url_input:
        if not st.session_state.api_key_valid:
            st.toast("⚠️ Please enter a valid OpenAI API key first!", icon="⚠️")
            st.error("❌ Please enter a valid OpenAI API key first")
        elif not url_input.strip():
            st.toast("⚠️ Please enter a website URL!", icon="⚠️")
            st.error("❌ Please enter a website URL")
        elif not st.session_state.assistant.is_valid_url(url_input.strip()):
            st.toast("⚠️ Please enter a valid URL!", icon="⚠️")
            st.error("❌ Please enter a valid URL (e.g., https://example.com)")
        else:
            # Show processing message
            with st.spinner("🔄 Processing website content..."):
                try:
                    # Extract content
                    content_data = st.session_state.assistant.extract_text_from_url(url_input.strip())

                    if not content_data['success']:
                        st.error(f"❌ Failed to process website: {content_data['error']}")
                        st.toast(f"❌ Failed to process website: {content_data['error']}", icon="❌")
                    else:
                        # Process content and create vector store
                        vectorstore = st.session_state.assistant.process_content(content_data)

                        # Create QA chain
                        qa_chain = st.session_state.assistant.create_qa_chain(vectorstore)

                        # Store in session state
                        st.session_state.processed_url = url_input.strip()
                        st.session_state.vectorstore = vectorstore
                        st.session_state.qa_chain = qa_chain
                        st.session_state.website_info = content_data
                        st.session_state.chat_history = []
                        st.session_state.selected_url = url_input.strip()

                        st.markdown(f"""
                        <div class="success-box">
                        ✅ <strong>Website '{content_data['title']}' processed successfully!</strong><br>
                        You can now ask questions about the content below.
                        </div>
                        """, unsafe_allow_html=True)

                        st.toast(f"✅ Website processed successfully!", icon="✅")

                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error processing content: {error_msg}")
                    st.toast(f"❌ Error: {error_msg}", icon="❌")
                    # Clear any partial state
                    for key in ['vectorstore', 'qa_chain', 'website_info']:
                        if key in st.session_state:
                            del st.session_state[key]

    # Q&A Interface
    if st.session_state.qa_chain and st.session_state.api_key_valid:
        st.header("💬 Ask Questions About the Website")

        # Chat interface
        question = st.text_input(
            "Enter your question:",
            placeholder="What is this website about?",
            key="question_input"
        )

        if st.button("🤔 Ask Question", key="ask_btn") and question:
            if not question.strip():
                st.warning("⚠️ Please enter a valid question.")
                st.toast("⚠️ Please enter a valid question!", icon="⚠️")
                return

            with st.spinner("🧠 Thinking..."):
                try:
                    # Check if API key is still valid
                    if not st.session_state.api_key_valid:
                        st.toast("⚠️ Please enter your OpenAI API key first!", icon="⚠️")
                        return

                    # Get answer from QA chain
                    result = st.session_state.qa_chain({"question": question.strip()})

                    # Extract answer safely
                    if isinstance(result, dict):
                        answer = result.get("answer", "Sorry, I couldn't generate an answer.")
                        source_docs = result.get("source_documents", [])
                    else:
                        answer = str(result)
                        source_docs = []

                    # Store in chat history
                    st.session_state.chat_history.append({
                        "question": question.strip(),
                        "answer": answer,
                        "source_documents": len(source_docs),
                        "timestamp": time.time()
                    })

                    # Display the answer immediately
                    st.success("✅ Answer generated successfully!")
                    st.toast("✅ Answer generated!", icon="✅")
                    with st.expander("📝 Latest Answer", expanded=True):
                        st.markdown(f"**Question:** {question.strip()}")
                        st.markdown(f"**Answer:** {answer}")
                        if source_docs:
                            st.caption(f"Based on {len(source_docs)} relevant document sections")

                    # Clear the input by refreshing
                    st.rerun()

                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error generating answer: {error_msg}")
                    st.toast(f"❌ Error generating answer: {error_msg}", icon="❌")

                    # Log the error for debugging
                    st.session_state.last_error = {
                        "error": error_msg,
                        "question": question,
                        "timestamp": time.time()
                    }

        # Display chat history
        if st.session_state.chat_history:
            st.header("📝 Conversation History")

            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{len(st.session_state.chat_history) - i}: {chat['question'][:50]}..."):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
                    st.markdown("---")

    elif url_input and not process_button:
        st.info("👆 Click 'Process Website' to analyze the content and start asking questions!")

    else:
        st.info("🌐 Enter a website URL above to get started!")

        # Demo section with example websites (matching welcome screen design)
        st.header("📚 Try These Example Websites")
        st.markdown("Click any example below to auto-fill the URL field:")

        # Create example URLs with descriptions (same as welcome screen)
        example_urls = [
            {
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "title": "🧠 Wikipedia - Artificial Intelligence",
                "description": "Learn about AI concepts and history"
            },
            {
                "url": "https://python.langchain.com/docs/get_started/introduction",
                "title": "🔗 LangChain Documentation",
                "description": "Explore LangChain framework docs"
            },
            {
                "url": "https://streamlit.io/",
                "title": "⚡ Streamlit Homepage",
                "description": "Discover Streamlit app framework"
            },
            {
                "url": "https://openai.com/",
                "title": "🤖 OpenAI Homepage",
                "description": "Learn about OpenAI and their AI models"
            },
            {
                "url": "https://docs.python.org/3/",
                "title": "🐍 Python Documentation",
                "description": "Official Python 3 documentation"
            },
            {
                "url": "https://github.com/",
                "title": "💻 GitHub Homepage",
                "description": "Explore the world's largest code repository"
            }
        ]

        # Display examples in a nice grid (same as welcome screen)
        example_cols = st.columns(2)
        for i, example in enumerate(example_urls):
            with example_cols[i % 2]:
                if st.button(
                        f"{example['title']}",
                        help=example['description'],
                        key=f"main_example_{i}",
                        use_container_width=True
                ):
                    # Set the URL for auto-processing
                    st.session_state.auto_process_url = example['url']
                    st.rerun()
                st.caption(example['description'])


def render_footer_sections():
    """Render the footer sections - always visible regardless of API key status"""

    # 1. Custom AI Solutions Contact Form
    st.markdown("---")
    with st.container():
        st.markdown("""
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: #333; margin-bottom: 0.5rem;">🤖 Interested in Custom AI Solutions?</h2>
            <h3 style="color: #666; margin-bottom: 2rem;">Let's discuss your AI needs</h3>
        </div>
        """, unsafe_allow_html=True)

        # Create the contact form
        with st.form("contact_form"):
            form_col1, form_col2 = st.columns(2)

            with form_col1:
                name = st.text_input("Name*", placeholder="Your full name", key="contact_name")
                email = st.text_input("Email*", placeholder="your@email.com", key="contact_email")

            with form_col2:
                company = st.text_input("Company", placeholder="Your company name", key="contact_company")
                phone = st.text_input("Phone", placeholder="+1 (555) 123-4567", key="contact_phone")

            message = st.text_area(
                "Tell us about your project*",
                placeholder="Describe your requirements, timeline, and any specific needs...",
                height=150,
                key="contact_message"
            )

            # Submit button
            submitted = st.form_submit_button("🚀 Send Message", type="primary")

            if submitted:
                # Validate required fields
                if not name or not email or not message:
                    st.error("Please fill in all required fields (marked with *)")
                    st.toast("⚠️ Please fill in all required fields!", icon="⚠️")
                else:
                    # Here you would typically send the form data to your backend
                    st.success("✅ Thank you for your interest! We'll get back to you within 24 hours.")
                    st.toast("✅ Message sent successfully!", icon="✅")
                    st.balloons()

                    # Store the form data (in a real app, this would go to a database or email service)
                    form_data = {
                        "name": name,
                        "email": email,
                        "company": company,
                        "phone": phone,
                        "message": message,
                        "timestamp": time.time()
                    }

                    # Log the submission (for demo purposes)
                    st.info(f"Form submitted by {name} from {company or 'N/A'}")

    # 2. Transform Your Business with AI Banner
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h3>🚀 Ready to Transform Your Business with AI?</h3>
            <p>This demo showcases intelligent website content understanding through natural language processing</p>
            <p><strong>Built with ❤️ by Nirix AI</strong> | Specializing in LangChain & Document Intelligence Solutions</p>
            <a href="mailto:ajaysharmabki96@gmail.com" style="color: white; text-decoration: none; margin: 0 1rem;">
                📧 ajaysharmabki96@gmail.com
            </a>
            <a href="tel:+919414256219" style="color: white; text-decoration: none; margin: 0 1rem;">
                📞 +91 9414256219
            </a>
            <a href="https://nirixai-agency.web.app/" target="_blank" style="color: white; text-decoration: none; margin: 0 1rem;">
                🌐 NirixAI Agency
            </a>
        </div>
        """, unsafe_allow_html=True)

    # 3. About This Demo
    st.markdown("---")
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        **Website Content Assistant** demonstrates advanced AI capabilities for website content understanding:

        **Key Features:**
        - **Multi-format Support:** Any public website content
        - **Intelligent Chunking:** Optimal text splitting for better retrieval
        - **Semantic Search:** Vector embeddings for context-aware answers
        - **Conversation Memory:** Maintains context across questions
        - **Source Attribution:** Shows relevant content sections
        - **Usage Tracking:** Token usage and cost monitoring

        **Technical Implementation:**
        - **LangChain:** Orchestrates the AI workflow
        - **OpenAI GPT:** Powers the question-answering
        - **FAISS:** Vector database for fast similarity search
        - **Streamlit:** Interactive web interface

        **Business Value:**
        - **Instant website understanding**
        - **Automated knowledge extraction**
        - **Improved information accessibility**
        - **Reduced manual content review time**
        """)

    # Original footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Built with LangChain</strong> | Powered by OpenAI | Made with ❤️ using Streamlit</p>
        <p>Transform any website into a conversational interface</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()