import streamlit as st
import time
import re
from typing import List, Dict, Tuple
import io
import PyPDF2
from datetime import datetime
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# Page configuration
st.set_page_config(
    page_title="Smart FAQ Generator | AI-Powered Document Processing",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }

    .contact-form {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }

    .transform-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }

    .demo-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }

    .faq-item {
        background: white;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .question {
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }

    .answer {
        color: #666;
        line-height: 1.6;
    }

    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }

    .stButton > button {
        background: #FF6B6B !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 5px !important;
        font-weight: bold !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: #FF5252 !important;
        color: white !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button:focus {
        background: #FF5252 !important;
        color: white !important;
        box-shadow: 0 0 0 2px rgba(255, 107, 107, 0.5) !important;
    }

    .disabled-section {
        opacity: 0.5;
        pointer-events: none;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 2px dashed #dee2e6;
    }

    .api-status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }

    .api-status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'sample_text_loaded' not in st.session_state:
    st.session_state.sample_text_loaded = False


def validate_openai_api_key(api_key: str) -> bool:
    """Simple API key format validation"""
    # Just check if it looks like an OpenAI API key format
    if api_key and api_key.startswith('sk-') and len(api_key) > 20:
        return True
    return False


def generate_faqs_with_langchain(text: str, api_key: str) -> List[Dict[str, str]]:
    """
    Generate FAQs using LangChain with OpenAI (Updated for LangChain v0.2+)
    """
    try:
        # Set the API key
        os.environ["OPENAI_API_KEY"] = api_key

        # Initialize the LLM
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1500
        )

        # Create the prompt template
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the following business document/text and generate 5-6 relevant FAQs with detailed answers based on the content.
            The questions should cover key aspects like services, products, processes, pricing, support, and unique features.

            Document Content:
            {text}

            Please respond with a JSON array in this exact format:
            [
                {{"question": "What services does the company provide?", "answer": "Detailed answer based on the content..."}},
                {{"question": "How does the process work?", "answer": "Detailed answer based on the content..."}}
            ]

            Make sure:
            1. Questions are relevant to the business/content provided
            2. Answers are comprehensive and directly based on the provided content
            3. Response is valid JSON format only
            4. Generate 5-6 FAQs maximum
            """
        )

        # Create the chain using the new LCEL syntax
        chain = prompt | llm

        # Run the chain using invoke method
        response = chain.invoke({"text": text})

        # Extract content from the response
        response_content = response.content if hasattr(response, 'content') else str(response)

        # Try to parse JSON response
        try:
            # Clean the response (remove any markdown formatting)
            clean_response = response_content.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response.replace("```json", "").replace("```", "")
            if clean_response.startswith("```"):
                clean_response = clean_response.replace("```", "")

            faqs = json.loads(clean_response)

            # Validate the structure
            if isinstance(faqs, list) and len(faqs) > 0:
                for faq in faqs:
                    if not isinstance(faq, dict) or 'question' not in faq or 'answer' not in faq:
                        raise ValueError("Invalid FAQ structure")
                return faqs
            else:
                raise ValueError("Invalid response format")

        except (json.JSONDecodeError, ValueError) as e:
            st.warning(f"⚠️ Could not parse AI response as JSON. Using fallback method. Error: {str(e)}")
            return generate_enhanced_mock_faqs(text)

    except Exception as e:
        st.error(f"Error generating FAQs with LangChain: {str(e)}")
        return generate_enhanced_mock_faqs(text)


def generate_enhanced_mock_faqs(text: str) -> List[Dict[str, str]]:
    """
    Fallback mock FAQ generation
    """
    # Extract key topics and generate relevant FAQs
    sentences = text.split('.')
    paragraphs = text.split('\n\n')

    # Mock intelligent FAQ generation based on content analysis
    faqs = []

    # Company/Service related FAQs
    if any(word in text.lower() for word in ['company', 'business', 'service', 'organization']):
        faqs.append({
            "question": "What services does your company provide?",
            "answer": "Based on the document, our company specializes in comprehensive solutions tailored to meet diverse client needs. We focus on delivering high-quality services with a customer-centric approach."
        })

    # Product related FAQs
    if any(word in text.lower() for word in ['product', 'solution', 'offering', 'feature']):
        faqs.append({
            "question": "What are the key features of your products?",
            "answer": "Our products are designed with cutting-edge technology and user-friendly interfaces. They incorporate advanced features that enhance productivity and deliver exceptional value to our customers."
        })

    # Process/Procedure related FAQs
    if any(word in text.lower() for word in ['process', 'procedure', 'step', 'method', 'how']):
        faqs.append({
            "question": "How does your process work?",
            "answer": "Our streamlined process ensures efficiency and transparency. We follow a systematic approach that includes consultation, planning, implementation, and ongoing support to guarantee optimal results."
        })

    # Contact/Support related FAQs
    if any(word in text.lower() for word in ['contact', 'support', 'help', 'assistance']):
        faqs.append({
            "question": "How can I get support or contact you?",
            "answer": "We provide multiple channels for support including email, phone, and online chat. Our dedicated support team is available to assist you with any questions or concerns you may have."
        })

    # Pricing/Cost related FAQs
    if any(word in text.lower() for word in ['price', 'cost', 'fee', 'payment', 'billing']):
        faqs.append({
            "question": "What are your pricing options?",
            "answer": "We offer flexible pricing structures to accommodate different needs and budgets. Our pricing is transparent and competitive, with various packages available to suit different requirements."
        })

    # Generic FAQs based on content
    if len(paragraphs) > 3:
        faqs.append({
            "question": "What makes your approach unique?",
            "answer": "Our unique approach combines industry expertise with innovative solutions. We prioritize customer satisfaction and continuous improvement to deliver exceptional results that exceed expectations."
        })

    # Ensure we have at least 3 FAQs
    if len(faqs) < 3:
        faqs.extend([
            {
                "question": "What can I expect from your service?",
                "answer": "You can expect professional, reliable service with attention to detail. We are committed to delivering high-quality results that meet your specific requirements and timeline."
            },
            {
                "question": "How do you ensure quality?",
                "answer": "Quality is our top priority. We implement rigorous quality control measures throughout our process, including regular reviews, testing, and feedback incorporation to ensure excellence."
            },
            {
                "question": "What is your typical delivery timeline?",
                "answer": "Our delivery timelines are tailored to each project's complexity and requirements. We work efficiently while maintaining high standards to ensure timely delivery without compromising quality."
            }
        ])

    return faqs[:6]  # Return maximum 6 FAQs


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def create_faq_download(faqs: List[Dict[str, str]]) -> str:
    """Create downloadable FAQ content"""
    content = "# Generated FAQs\n\n"
    for i, faq in enumerate(faqs, 1):
        content += f"## {i}. {faq['question']}\n\n"
        content += f"{faq['answer']}\n\n"
        content += "---\n\n"

    content += f"\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    content += "*Powered by Nirix AI - LangChain & Document Intelligence Solutions*"

    return content


def main():
    # Sidebar for API Configuration
    with st.sidebar:
        st.markdown("## 🔑 API Configuration")
        st.markdown("Enter your OpenAI API Key:")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key to enable AI-powered FAQ generation"
        )

        if api_key:
            if st.button("Validate API Key"):
                if validate_openai_api_key(api_key):
                    st.session_state.api_key_validated = True
                    st.session_state.openai_api_key = api_key
                    st.markdown("""
                    <div class="api-status-success">
                        ✅ API key format validated successfully!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.session_state.api_key_validated = False
                    st.markdown("""
                    <div class="api-status-error">
                        ❌ Invalid API key format. Please check and try again.
                    </div>
                    """, unsafe_allow_html=True)

        if not api_key:
            st.markdown("""
            <div style="background: #fff3cd; color: #856404; padding: 0.75rem; border-radius: 5px; border: 1px solid #ffeaa7; margin: 0.5rem 0;">
                ⚠️ Please enter your OpenAI API key to continue
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📖 How to Use")
        st.markdown("""
        **Step 1:** Enter your OpenAI API key

        **Step 2:** Upload PDF or paste text content

        **Step 3:** Click "Generate Smart FAQs"

        **Step 4:** Review generated questions & answers

        **Step 5:** Download FAQs in your preferred format
        """)

    # Header Section
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Smart FAQ Generator</h1>
        <p>Transform your documents into smart FAQs automatically</p>
        <p><em>AI-Powered Document Processing with LangChain</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Check if API key is validated
    api_enabled = st.session_state.get('api_key_validated', False)

    if not api_enabled:
        st.markdown("""
        <div class="disabled-section">
            <h2>📄 Generate FAQs from Your Documents</h2>
            <p style="text-align: center; color: #6c757d; font-size: 1.1rem;">
                🔒 Please configure your OpenAI API key in the sidebar to enable FAQ generation
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Main FAQ Generation Section
        st.markdown("## 📄 Generate FAQs from Your Documents")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Upload Your Document")

            # File upload options
            upload_option = st.radio(
                "Choose input method:",
                ["Upload PDF", "Paste Text"]
            )

            text_content = ""

            if upload_option == "Upload PDF":
                uploaded_file = st.file_uploader(
                    "Upload a PDF document",
                    type=['pdf'],
                    help="Upload a PDF containing company information, policies, or documentation"
                )

                if uploaded_file is not None:
                    with st.spinner("Extracting text from PDF..."):
                        text_content = extract_text_from_pdf(uploaded_file)
                        if text_content and len(text_content.strip()) > 0:
                            st.success(f"✅ PDF processed successfully! Extracted {len(text_content)} characters.")
                            st.text_area("Extracted Text Preview:", value=text_content[:500] + "...", height=100,
                                         disabled=True)
                        else:
                            st.error("❌ Could not extract text from PDF. Please try a different file.")

            else:  # Paste Text
                # Check if sample text was loaded
                if 'sample_text_loaded' in st.session_state and st.session_state.sample_text_loaded:
                    default_text = st.session_state.get('sample_text', '')
                    st.session_state.sample_text_loaded = False  # Reset flag
                else:
                    default_text = ""

                text_content = st.text_area(
                    "Paste your company information, policies, or documentation:",
                    height=200,
                    placeholder="Enter your company information, product details, policies, or any documentation you want to convert into FAQs...",
                    value=default_text,
                    key="text_input"
                )

        with col2:
            st.markdown("### 🚀 Quick Demo")
            st.markdown("""
            **Try with sample text:**
            - Company information
            - Product documentation
            - Policy documents
            - Service descriptions
            - Process guidelines
            """)

            if st.button("Use Sample Company Info", key="sample_button"):
                sample_text = """
ABC Technology Solutions is a leading software development company specializing in AI and machine learning solutions. 
We provide custom software development, AI consulting, and digital transformation services to businesses worldwide.

Our services include:
- Custom AI application development
- Machine learning model training
- Data analytics and visualization
- Cloud migration and optimization
- Technical consulting and support

We follow an agile development methodology with continuous integration and deployment practices. 
Our team consists of experienced engineers, data scientists, and AI specialists.

Pricing is project-based and depends on scope and complexity. We offer free initial consultations 
and provide detailed project estimates. Support is available 24/7 through our dedicated helpdesk.

Contact us at info@abctech.com or call +1-555-123-4567 for more information.
                """
                # Store sample text in session state
                st.session_state.sample_text = sample_text
                st.session_state.sample_text_loaded = True
                st.rerun()

        # Handle sample text loading
        if st.session_state.get('sample_text_loaded', False) and upload_option == "Paste Text":
            text_content = st.session_state.get('sample_text', '')

        # Debug information
        if text_content:
            st.info(f"📊 Content length: {len(text_content)} characters")

        # Generate FAQs Button - Show when text content is available
        if text_content and len(text_content.strip()) > 100:
            st.markdown("---")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Generate Smart FAQs", use_container_width=True, key="generate_faqs"):
                    with st.spinner("🧠 AI is analyzing your content and generating FAQs..."):
                        # Use LangChain with OpenAI if API key is available
                        if api_enabled and 'openai_api_key' in st.session_state:
                            faqs = generate_faqs_with_langchain(text_content, st.session_state.openai_api_key)
                        else:
                            time.sleep(2)  # Simulate processing time
                            faqs = generate_enhanced_mock_faqs(text_content)

                        st.session_state['generated_faqs'] = faqs
                        st.success("✅ FAQs generated successfully!")

        elif text_content and len(text_content.strip()) <= 100:
            st.warning("⚠️ Please provide more content (minimum 100 characters) for better FAQ generation.")

        # Display Generated FAQs
        if 'generated_faqs' in st.session_state:
            st.markdown("---")
            st.markdown("## 📋 Generated FAQs")

            faqs = st.session_state['generated_faqs']

            for i, faq in enumerate(faqs, 1):
                st.markdown(f"""
                <div class="faq-item">
                    <div class="question">Q{i}: {faq['question']}</div>
                    <div class="answer">{faq['answer']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Download Options
            st.markdown("### 📥 Export Options")
            col1, col2 = st.columns(2)

            with col1:
                faq_content = create_faq_download(faqs)
                st.download_button(
                    label="📄 Download as Markdown",
                    data=faq_content,
                    file_name=f"generated_faqs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

            with col2:
                json_content = json.dumps(faqs, indent=2)
                st.download_button(
                    label="📊 Download as JSON",
                    data=json_content,
                    file_name=f"generated_faqs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    # 1. Contact Form Section - Interested in Custom AI Solutions
    st.markdown("---")
    st.markdown("## 💼 Interested in Custom AI Solutions?")
    st.markdown("### Let's discuss your AI needs")

    with st.form("contact_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Name*", placeholder="Your full name")
            email = st.text_input("Email*", placeholder="your@email.com")

        with col2:
            company = st.text_input("Company", placeholder="Your company name")
            phone = st.text_input("Phone", placeholder="+1 (555) 123-4567")

        project_description = st.text_area(
            "Tell us about your project*",
            placeholder="Describe your requirements, timeline, and any specific needs...",
            height=100
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button("🚀 Send Message", use_container_width=True)

        if submitted:
            if name and email and project_description:
                st.markdown("""
                <div class="success-message">
                    <strong>✅ Thank you for your interest!</strong><br>
                    We'll get back to you within 24 hours to discuss your AI project needs.
                </div>
                """, unsafe_allow_html=True)

                # In production, you would save this to a database or send an email
                st.info(
                    "💡 **Next Steps:** Our AI specialists will review your requirements and schedule a consultation call.")
            else:
                st.error("⚠️ Please fill in all required fields (Name, Email, and Project Description)")

    # 2. Transform Your Business Section
    st.markdown("---")
    st.markdown("""
    <div class="transform-banner">
        <h2>🚀 Ready to Transform Your Business with AI?</h2>
        <p>This demo showcases intelligent document processing through natural language understanding</p>
        <p><strong>Built with ❤️ by Nirix AI | Specializing in LangChain & Document Intelligence Solutions</strong></p>
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

    # 3. About This Demo Section
    st.markdown("---")
    with st.expander("ℹ️ About This Demo", expanded=False):
        st.markdown("""
        ### Smart FAQ Generator

        This demo showcases advanced AI capabilities for automated content processing and FAQ generation.

        **Key Features:**
        - **Multi-format Support**: PDF and text document processing
        - **Intelligent Processing**: Optimal question extraction and answer generation
        - **Contextual Understanding**: AI-powered content comprehension
        - **Export Options**: Download generated FAQs in multiple formats
        - **Real-time Results**: Instant processing and display
        - **Source Attribution**: Answers based on provided content

        **Technical Implementation:**
        - **LangChain**: Orchestrates the AI workflow
        - **OpenAI GPT**: Powers the question-answering capabilities
        - **Streamlit**: Interactive web interface
        - **Text Processing**: Advanced document parsing and analysis
        - **Vector Search**: Context-aware content retrieval

        **Business Value:**
        - **Automated knowledge extraction** from documents
        - **Reduced manual content creation time**
        - **Improved information accessibility**
        - **Scalable content processing**
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Smart FAQ Generator</strong> - Powered by LangChain & Advanced AI</p>
        <p>© 2024 Nirix AI. Specializing in Document Intelligence & AI Solutions</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()