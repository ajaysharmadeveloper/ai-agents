import streamlit as st
import os
import json
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# OpenAI imports
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.callbacks.manager import get_openai_callback
except ImportError:
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks import get_openai_callback

# Set page config
st.set_page_config(
    page_title="AI Product Discovery Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #e91e63;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
.product-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1rem;
    border-radius: 15px;
    margin: 1rem 0;
    border: 1px solid #e0e0e0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.product-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.chat-message {
    padding: 1rem;
    border-radius: 15px;
    margin: 1rem 0;
    max-width: 80%;
}
.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: auto;
    text-align: right;
}
.assistant-message {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
}
.price-tag {
    font-size: 1.2rem;
    font-weight: bold;
    color: #2e7d32;
}
.rating-stars {
    color: #ffa726;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)


@dataclass
class Product:
    id: str
    name: str
    category: str
    price: float
    description: str
    features: List[str]
    rating: float
    reviews_count: int
    brand: str
    image_url: str = ""
    tags: List[str] = None


def initialize_session_state():
    """Initialize session state variables"""
    if 'products' not in st.session_state:
        st.session_state.products = create_sample_products()
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'recommended_products' not in st.session_state:
        st.session_state.recommended_products = []
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {}


def create_sample_products() -> List[Product]:
    """Create sample e-commerce products"""
    products = [
        Product(
            id="1", name="MacBook Pro 16-inch", category="Electronics",
            price=2499.99,
            description="Powerful laptop with M2 Pro chip, perfect for professionals and creatives. Features stunning Retina display and all-day battery life.",
            features=["M2 Pro chip", "16-inch Retina display", "16GB RAM", "512GB SSD", "All-day battery"],
            rating=4.8, reviews_count=1250, brand="Apple",
            tags=["professional", "creative", "laptop", "powerful", "premium"]
        ),
        Product(
            id="2", name="Wireless Noise-Canceling Headphones", category="Electronics",
            price=299.99,
            description="Premium wireless headphones with industry-leading noise cancellation. Perfect for travel, work, and music enjoyment.",
            features=["Active noise cancellation", "30-hour battery", "Quick charge", "Bluetooth 5.0",
                      "Touch controls"],
            rating=4.6, reviews_count=892, brand="Sony",
            tags=["wireless", "music", "travel", "noise-canceling", "premium"]
        ),
        Product(
            id="3", name="Smart Fitness Watch", category="Wearables",
            price=399.99,
            description="Advanced fitness tracking with heart rate monitoring, GPS, and smart notifications. Water-resistant design for all activities.",
            features=["Heart rate monitor", "GPS tracking", "Water resistant", "Sleep tracking", "Smart notifications"],
            rating=4.4, reviews_count=2100, brand="Garmin",
            tags=["fitness", "health", "sports", "tracking", "smartwatch"]
        ),
        Product(
            id="4", name="Gaming Mechanical Keyboard", category="Gaming",
            price=149.99,
            description="RGB mechanical keyboard designed for gamers. Features tactile switches and customizable lighting effects.",
            features=["Mechanical switches", "RGB lighting", "Programmable keys", "Gaming mode", "USB-C connection"],
            rating=4.7, reviews_count=567, brand="Razer",
            tags=["gaming", "mechanical", "RGB", "keyboard", "esports"]
        ),
        Product(
            id="5", name="Portable Coffee Maker", category="Kitchen",
            price=89.99,
            description="Compact espresso machine perfect for small kitchens and offices. Makes barista-quality coffee in minutes.",
            features=["15-bar pressure", "Compact design", "Easy cleanup", "Multiple cup sizes", "Quick heating"],
            rating=4.3, reviews_count=445, brand="Nespresso",
            tags=["coffee", "compact", "office", "espresso", "convenient"]
        ),
        Product(
            id="6", name="Ergonomic Office Chair", category="Furniture",
            price=299.99,
            description="Professional office chair with lumbar support and adjustable features. Designed for long work sessions.",
            features=["Lumbar support", "Adjustable height", "Breathable mesh", "360° swivel", "Armrest adjustment"],
            rating=4.5, reviews_count=778, brand="Herman Miller",
            tags=["office", "ergonomic", "comfort", "work", "professional"]
        ),
        Product(
            id="7", name="4K Action Camera", category="Electronics",
            price=199.99,
            description="Waterproof action camera with 4K video recording. Perfect for adventures and sports activities.",
            features=["4K video", "Waterproof", "Image stabilization", "Wide angle lens", "WiFi connectivity"],
            rating=4.4, reviews_count=623, brand="GoPro",
            tags=["camera", "4K", "waterproof", "adventure", "sports"]
        ),
        Product(
            id="8", name="Smart Home Security System", category="Home Security",
            price=249.99,
            description="Complete home security system with cameras, sensors, and smartphone app control.",
            features=["HD cameras", "Motion detection", "Smartphone app", "Night vision", "Cloud storage"],
            rating=4.6, reviews_count=934, brand="Ring",
            tags=["security", "smart home", "cameras", "monitoring", "safety"]
        ),
        Product(
            id="9", name="Yoga Mat Premium", category="Fitness",
            price=49.99,
            description="High-quality yoga mat with excellent grip and cushioning. Made from eco-friendly materials.",
            features=["Non-slip surface", "6mm thickness", "Eco-friendly", "Easy to clean", "Carrying strap"],
            rating=4.5, reviews_count=1567, brand="Manduka",
            tags=["yoga", "fitness", "eco-friendly", "exercise", "meditation"]
        ),
        Product(
            id="10", name="Wireless Charging Pad", category="Electronics",
            price=39.99,
            description="Fast wireless charging pad compatible with all Qi-enabled devices. Sleek design for any desk or nightstand.",
            features=["Qi wireless charging", "Fast charging", "LED indicator", "Non-slip base",
                      "Universal compatibility"],
            rating=4.2, reviews_count=445, brand="Anker",
            tags=["wireless", "charging", "convenient", "universal", "fast"]
        )
    ]
    return products


def create_product_documents(products: List[Product]) -> List[Document]:
    """Convert products to documents for vector search"""
    documents = []
    for product in products:
        # Create rich product description for better semantic search
        content = f"""
        Product: {product.name}
        Brand: {product.brand}
        Category: {product.category}
        Price: ${product.price}
        Rating: {product.rating}/5 ({product.reviews_count} reviews)

        Description: {product.description}

        Key Features: {', '.join(product.features)}

        Tags: {', '.join(product.tags or [])}

        This product is great for: {', '.join(product.tags or [])}
        """

        doc = Document(
            page_content=content,
            metadata={
                'product_id': product.id,
                'name': product.name,
                'category': product.category,
                'price': product.price,
                'rating': product.rating,
                'brand': product.brand
            }
        )
        documents.append(doc)

    return documents


def setup_vector_search():
    """Initialize vector search for products"""
    if st.session_state.vectorstore is None:
        products = st.session_state.products
        documents = create_product_documents(products)

        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )
            vectorstore = FAISS.from_documents(documents, embeddings)
            st.session_state.vectorstore = vectorstore
            return True
        except Exception as e:
            st.error(f"Error setting up vector search: {str(e)}")
            return False
    return True


def create_conversation_chain():
    """Create conversation chain for product discovery"""
    if st.session_state.conversation is None:
        try:
            llm = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo",
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )

            # Custom prompt for product discovery
            template = """
            You are an expert AI shopping assistant helping customers find products. You have access to a product catalog and should:

            1. Ask clarifying questions to understand customer needs
            2. Recommend relevant products based on their requirements
            3. Explain why certain products match their needs
            4. Provide helpful comparisons between products
            5. Be conversational and friendly

            Current conversation:
            {history}

            Customer: {input}
            AI Assistant: """

            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=template
            )

            memory = ConversationBufferWindowMemory(
                k=10,  # Remember last 10 exchanges
                return_messages=False
            )

            conversation = ConversationChain(
                llm=llm,
                prompt=prompt,
                memory=memory,
                verbose=True
            )

            st.session_state.conversation = conversation
            return True
        except Exception as e:
            st.error(f"Error creating conversation chain: {str(e)}")
            return False
    return True


def search_products(query: str, k: int = 5) -> List[Product]:
    """Search products using semantic similarity"""
    if st.session_state.vectorstore is None:
        return []

    try:
        # Perform similarity search
        docs = st.session_state.vectorstore.similarity_search(query, k=k)

        # Get product IDs from search results
        product_ids = [doc.metadata['product_id'] for doc in docs]

        # Return corresponding products
        products = st.session_state.products
        relevant_products = [p for p in products if p.id in product_ids]

        return relevant_products
    except Exception as e:
        st.error(f"Error searching products: {str(e)}")
        return []


def display_product_card(product: Product):
    """Display a product in a card format"""
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"### {product.name}")
        st.caption(f"{product.brand} • {product.category}")
        st.write(product.description)

        # Display key features
        st.markdown("**Key Features:**")
        for feature in product.features[:3]:
            st.markdown(f"• {feature}")

    with col2:
        st.markdown(f"### ${product.price}")
        st.markdown(f"{'⭐' * int(product.rating)} {product.rating}/5")
        st.caption(f"({product.reviews_count} reviews)")

    st.markdown("---")


def handle_user_input(user_message: str):
    """Handle user input and provide product recommendations"""
    if not st.session_state.conversation:
        st.error("Please enter your OpenAI API key first!")
        return

    try:
        # Search for relevant products
        relevant_products = search_products(user_message)
        st.session_state.recommended_products = relevant_products

        # Create context with product information
        product_context = ""
        if relevant_products:
            product_context = "\n\nBased on your request, here are some relevant products:\n"
            for product in relevant_products[:3]:  # Top 3 products
                product_context += f"- {product.name} by {product.brand} (${product.price}) - {product.description}\n"

        # Get AI response
        with get_openai_callback() as cb:
            enhanced_message = user_message + product_context
            response = st.session_state.conversation.predict(input=enhanced_message)

            # Add to chat history
            st.session_state.chat_history.append({
                'user': user_message,
                'assistant': response,
                'products': relevant_products,
                'timestamp': datetime.now(),
                'tokens': cb.total_tokens if hasattr(cb, 'total_tokens') else 0
            })

    except Exception as e:
        st.error(f"Error processing your message: {str(e)}")


def display_chat_history():
    """Display chat history with product recommendations"""
    for chat in st.session_state.chat_history:
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {chat['user']}
        </div>
        """, unsafe_allow_html=True)

        # Assistant message
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>AI Assistant:</strong> {chat['assistant']}
        </div>
        """, unsafe_allow_html=True)

        # Product recommendations
        if chat['products']:
            st.markdown("**💡 Recommended Products:**")

            # Create a container with custom styling for product cards
            with st.container():
                # Display products in a grid layout
                for i in range(0, len(chat['products'][:3]), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(chat['products'][:3]):
                            product = chat['products'][i + j]
                            with col:
                                # Create a styled container for each product
                                with st.container():
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                            padding: 1rem;
                                            border-radius: 10px;
                                            border: 1px solid #e0e0e0;
                                            height: 100%;
                                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                        ">
                                            <h4 style="color: #1976d2; margin: 0 0 0.5rem 0; font-size: 1.1rem;">
                                                {product.name}
                                            </h4>
                                            <p style="color: #666; font-size: 0.9rem; margin: 0;">
                                                {product.brand} • {product.category}
                                            </p>
                                            <p style="color: #2e7d32; font-weight: bold; font-size: 1.2rem; margin: 0.5rem 0;">
                                                ${product.price}
                                            </p>
                                            <p style="color: #ffa726; margin: 0;">
                                                {'★' * int(product.rating)} {product.rating}/5
                                            </p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )


def main():
    """Main application function"""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">🛍️ AI Product Discovery Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Find exactly what you need through natural conversation</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # API Key
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

            # Initialize AI components
            if setup_vector_search() and create_conversation_chain():
                st.success("✅ AI Assistant Ready!")

        st.header("📊 Stats")
        st.metric("Products in Catalog", len(st.session_state.products))
        st.metric("Conversations", len(st.session_state.chat_history))

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.conversation = None
            st.rerun()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chat with AI Assistant")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            display_chat_history()

        # User input
        st.markdown("---")
        user_input = st.text_input(
            "💭 Tell me what you're looking for:",
            placeholder="I need a laptop for video editing...",
            key="user_input"
        )

        col_send, col_clear = st.columns([3, 1])
        with col_send:
            if st.button("🚀 Send Message", type="primary") and user_input:
                handle_user_input(user_input)
                st.rerun()

        # Sample questions
        with st.expander("💡 Try these example searches"):
            examples = [
                "I need a laptop for video editing under $3000",
                "Looking for wireless headphones for my daily commute",
                "Want a fitness tracker that can monitor my sleep",
                "Need a coffee maker for my small office",
                "Looking for gaming accessories under $200",
                "Want something to help me work from home comfortably"
            ]

            for example in examples:
                if st.button(f"💬 {example}", key=f"example_{example}"):
                    handle_user_input(example)
                    st.rerun()

    with col2:
        st.subheader("🏷️ Product Categories")

        # Category filter
        categories = list(set([p.category for p in st.session_state.products]))
        selected_category = st.selectbox("Filter by Category", ["All"] + categories)

        # Display products by category
        if selected_category == "All":
            filtered_products = st.session_state.products
        else:
            filtered_products = [p for p in st.session_state.products if p.category == selected_category]

        st.write(f"**{len(filtered_products)} products found**")

        # Display products in compact form
        for product in filtered_products[:5]:  # Show top 5
            with st.expander(f"{product.name} - ${product.price}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Brand:** {product.brand}")
                    st.write(f"**Description:** {product.description}")
                with col2:
                    st.metric("Rating", f"{product.rating}/5")
                    st.caption(f"{product.reviews_count} reviews")

                if st.button(f"Ask about {product.name}", key=f"ask_{product.id}"):
                    handle_user_input(f"Tell me more about the {product.name}")
                    st.rerun()

    # Custom AI Solutions Contact Form
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
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Name*", placeholder="Your full name")
                email = st.text_input("Email*", placeholder="your@email.com")

            with col2:
                company = st.text_input("Company", placeholder="Your company name")
                phone = st.text_input("Phone", placeholder="+1 (555) 123-4567")

            message = st.text_area(
                "Tell us about your project*",
                placeholder="Describe your requirements, timeline, and any specific needs...",
                height=150
            )

            # Submit button
            submitted = st.form_submit_button("🚀 Send Message", type="primary")

            if submitted:
                # Validate required fields
                if not name or not email or not message:
                    st.error("Please fill in all required fields (marked with *)")
                else:
                    # Here you would typically send the form data to your backend
                    st.success("✅ Thank you for your interest! We'll get back to you within 24 hours.")
                    st.balloons()

                    # Store the form data (in a real app, this would go to a database or email service)
                    form_data = {
                        "name": name,
                        "email": email,
                        "company": company,
                        "phone": phone,
                        "message": message,
                        "timestamp": datetime.now()
                    }

                    # Log the submission (for demo purposes)
                    st.info(f"Form submitted by {name} from {company or 'N/A'}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
        <h3>🚀 Transform Your E-commerce with AI</h3>
        <p>This demo showcases intelligent product discovery through natural language processing</p>
        <p><strong>Built by Nirix AI</strong> | Specializing in LangChain & E-commerce AI Solutions</p>
        <p>📧 ajaysharmabki96@gmail.com | 📞 +91 9414256219 | 🌐 nirixai.com</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()