# AI Product Discovery Assistant - Setup & Deployment Guide

## 🛍️ Overview

An intelligent e-commerce chatbot that helps customers find products through natural conversation. Built with LangChain, OpenAI, and Streamlit to demonstrate advanced AI-powered product discovery capabilities.

## ✨ Features

- **Natural Language Search**: Describe what you need in plain English
- **Semantic Product Matching**: AI-powered vector search for relevant products
- **Conversational Memory**: Maintains context throughout the shopping session
- **Smart Recommendations**: Suggests products based on customer requirements
- **Category Filtering**: Browse products by category
- **Interactive Product Cards**: Beautiful product display with ratings and features
- **Real-time Chat Interface**: Seamless conversation flow

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Create project directory
mkdir ai-product-discovery
cd ai-product-discovery

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
STREAMLIT_SERVER_PORT=8502
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 3. Run the Application
```bash
streamlit run product_discovery_app.py
```

## 📦 Requirements

Create a `requirements.txt` file with:
```text
streamlit
langchain
langchain-openai
langchain-community
openai
faiss-cpu
tiktoken
pandas
numpy
python-dotenv
```

## 🏗️ Project Structure

```
ai-product-discovery/
├── product_discovery_app.py   # Main application file
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── .env                      # Environment variables
└── .gitignore               # Git ignore file
```

## 🔧 Configuration

### OpenAI API Key
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add it to your `.env` file or enter it in the app sidebar
3. Ensure you have sufficient credits for embeddings and chat completions

### Customization Options
- **Product Catalog**: Modify the `create_sample_products()` function to add your products
- **UI Styling**: Update CSS in the `st.markdown()` sections
- **AI Behavior**: Adjust the conversation prompt template
- **Search Parameters**: Modify vector search parameters for different results

## 🎯 Usage Examples

Try these example queries:
- "I need a laptop for video editing under $3000"
- "Looking for wireless headphones for my daily commute"
- "Want a fitness tracker that can monitor my sleep"
- "Need a coffee maker for my small office"
- "Looking for gaming accessories under $200"

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your `OPENAI_API_KEY` in the app secrets
4. Deploy with one click

### Local Network Access
```bash
streamlit run product_discovery_app.py --server.address 0.0.0.0 --server.port 8502
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8502

CMD ["streamlit", "run", "product_discovery_app.py", "--server.address", "0.0.0.0", "--server.port", "8502"]
```

## 🛠️ Technical Architecture

### Core Components
- **LangChain**: Orchestrates the AI workflow
- **OpenAI GPT-3.5**: Powers conversational AI
- **FAISS Vector Store**: Enables semantic product search
- **Streamlit**: Provides interactive web interface

### Data Flow
1. User enters natural language query
2. AI assistant processes the request
3. Vector search finds relevant products
4. LLM generates contextual response
5. Products displayed with recommendations

## 🎨 Customization

### Adding New Products
```python
# In create_sample_products() function
Product(
    id="new_id",
    name="Product Name",
    category="Category",
    price=99.99,
    description="Product description",
    features=["Feature 1", "Feature 2"],
    rating=4.5,
    reviews_count=100,
    brand="Brand Name",
    tags=["tag1", "tag2"]
)
```

### Modifying AI Behavior
```python
# Update the conversation template
template = """
Your custom AI assistant prompt here...
{history}
Customer: {input}
AI Assistant: """
```

## 🔍 Troubleshooting

### Common Issues
1. **OpenAI API Error**: Verify your API key and account credits
2. **Import Errors**: Ensure all dependencies are installed with correct versions
3. **Vector Search Issues**: Check if FAISS is properly installed
4. **Memory Issues**: Restart the app to clear conversation memory

### Debug Mode
Add this to your app for debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Analytics & Monitoring

The app tracks:
- Number of conversations
- Product recommendations made
- Token usage and costs
- User interaction patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

**Built by Nirix AI** - Specializing in LangChain & E-commerce AI Solutions

- 📧 Email: ajaysharmabjki96@gmail.com
- 📞 Phone: +91 9414256219
- 🌐 Website: nirixai.com

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Business Applications

### Use Cases
- **E-commerce Product Discovery**: Help customers find products naturally
- **Customer Support**: Answer product-related questions
- **Personalized Shopping**: Tailor recommendations to user preferences
- **Inventory Management**: Track popular product searches

### ROI Benefits
- **Increased Conversion**: Better product discovery leads to more sales
- **Reduced Support Costs**: Automated product assistance
- **Enhanced User Experience**: Natural language shopping interface
- **Data Insights**: Understand customer search patterns

---

*This is a demonstration project showcasing AI-powered e-commerce capabilities. Perfect for client presentations and proof-of-concept demonstrations.*