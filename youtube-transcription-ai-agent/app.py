import streamlit as st
import os
import tempfile
import re
from typing import List, Dict, Any
import logging

# Fix OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Core libraries
import yt_dlp
import whisper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeTranscriptionApp:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="YouTube AI Chatbot",
            page_icon="🎥",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'transcription' not in st.session_state:
            st.session_state.transcription = ""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'video_info' not in st.session_state:
            st.session_state.video_info = {}
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None

    def validate_youtube_url(self, url: str) -> bool:
        """Validate if the URL is a valid YouTube URL"""
        youtube_regex = re.compile(
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        return bool(youtube_regex.match(url))

    def extract_video_info(self, url: str) -> Dict[str, Any]:
        """Extract video metadata using yt-dlp"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', ''),
                    'thumbnail': info.get('thumbnail', ''),
                    'video_id': info.get('id', '')
                }
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            return {}

    def download_audio(self, url: str) -> str:
        """Download audio from YouTube video"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                # Find the downloaded file
                for file in os.listdir(temp_dir):
                    if file.endswith('.mp3'):
                        audio_path = os.path.join(temp_dir, file)
                        # Copy to a persistent location
                        permanent_path = os.path.join(os.getcwd(), 'youtube-transcription-ai-agent/temp_audio.mp3')
                        os.rename(audio_path, permanent_path)
                        return permanent_path

        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Load model with CPU-optimized settings
            model = whisper.load_model("base", device="cpu")
            result = model.transcribe(
                audio_path,
                fp16=False,  # Disable FP16 for CPU compatibility
                verbose=False  # Reduce output
            )
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            st.error(f"Transcription failed: {str(e)}")
            return ""
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass  # Ignore cleanup errors

    def setup_qa_chain(self, openai_api_key: str, transcription: str, video_info: Dict):
        """Set up the QA chain with vector store for the transcription"""
        try:
            # Create documents from transcription
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            # Add video metadata to the context
            context = f"""
            Video Title: {video_info.get('title', 'Unknown')}
            Uploader: {video_info.get('uploader', 'Unknown')}
            Duration: {video_info.get('duration', 0)} seconds
            Description: {video_info.get('description', '')[:500]}...

            Transcription:
            {transcription}
            """

            texts = text_splitter.split_text(context)
            documents = [Document(page_content=text) for text in texts]

            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            vector_store = FAISS.from_documents(documents, embeddings)

            # Create conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Create the LLM with modern ChatOpenAI
            llm = ChatOpenAI(
                api_key=openai_api_key,
                temperature=0.7,
                model="gpt-3.5-turbo"
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                return_source_documents=True,
                verbose=False  # Reduce verbosity to prevent issues
            )

            return qa_chain, vector_store

        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            st.error(f"Failed to setup QA chain: {str(e)}")
            return None, None

    def render_sidebar(self):
        """Render the sidebar with API key input"""
        st.sidebar.header("🔑 Configuration")

        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Enter your OpenAI API key...",
            help="Your API key is used locally and not stored anywhere."
        )

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.sidebar.success("✅ API Key configured!")
            return api_key
        else:
            st.sidebar.warning("⚠️ Please enter your OpenAI API key to continue.")
            return None

    def render_video_section(self, api_key: str):
        """Render the video input and processing section"""
        st.header("🎥 YouTube Video Transcription")

        # Video URL input
        video_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL here"
        )

        if video_url and self.validate_youtube_url(video_url):
            col1, col2 = st.columns([3, 1])

            with col2:
                process_button = st.button("🚀 Process Video", type="primary")

            if process_button:
                with st.spinner("Processing video..."):
                    # Extract video info
                    st.info("📊 Extracting video information...")
                    video_info = self.extract_video_info(video_url)

                    if video_info:
                        st.session_state.video_info = video_info

                        # Download audio
                        st.info("🎵 Downloading audio...")
                        audio_path = self.download_audio(video_url)

                        if audio_path:
                            # Transcribe audio
                            st.info("🎤 Transcribing audio...")
                            transcription = self.transcribe_audio(audio_path)

                            if transcription:
                                st.session_state.transcription = transcription

                                # Setup QA chain
                                st.info("🤖 Setting up AI assistant...")
                                qa_chain, vector_store = self.setup_qa_chain(
                                    api_key, transcription, video_info
                                )

                                if qa_chain:
                                    st.session_state.qa_chain = qa_chain
                                    st.session_state.vector_store = vector_store
                                    st.success("✅ Video processed successfully!")
                                else:
                                    st.error("❌ Failed to setup AI assistant")
                            else:
                                st.error("❌ Failed to transcribe audio")
                        else:
                            st.error("❌ Failed to download audio")
                    else:
                        st.error("❌ Failed to extract video information")

        elif video_url and not self.validate_youtube_url(video_url):
            st.error("❌ Please enter a valid YouTube URL")

    def render_video_player(self):
        """Render the video player and info"""
        if st.session_state.video_info:
            info = st.session_state.video_info

            # Display video
            st.subheader("📺 Video Player")
            video_id = info.get('video_id', '')
            if video_id:
                st.video(f"https://www.youtube.com/watch?v={video_id}")

            # Display video information
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Title:** {info.get('title', 'Unknown')}")
                st.write(f"**Uploader:** {info.get('uploader', 'Unknown')}")
                st.write(f"**Duration:** {info.get('duration', 0)} seconds")

            with col2:
                st.write(f"**Views:** {info.get('view_count', 0):,}")
                st.write(f"**Upload Date:** {info.get('upload_date', 'Unknown')}")

            # Show transcription
            if st.session_state.transcription:
                with st.expander("📝 View Transcription"):
                    st.text_area(
                        "Transcription:",
                        st.session_state.transcription,
                        height=200,
                        disabled=True
                    )

    def render_chat_interface(self):
        """Render the AI chat interface"""
        if st.session_state.qa_chain and st.session_state.transcription:
            st.header("💬 AI Assistant Chat")
            st.write("Ask me anything about the video content!")

            # Display chat history
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)

            # Chat input
            if prompt := st.chat_input("Ask about the video..."):
                # Add user message to chat history
                with st.chat_message("user"):
                    st.write(prompt)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Use invoke instead of __call__ for modern LangChain
                            response = st.session_state.qa_chain.invoke({
                                "question": prompt,
                                "chat_history": []  # Reset for each question to avoid memory issues
                            })
                            answer = response["answer"]
                            st.write(answer)

                            # Add to chat history
                            st.session_state.chat_history.append((prompt, answer))

                        except Exception as e:
                            error_msg = f"Error processing your question: {str(e)}"
                            st.error(error_msg)
                            logger.error(f"Chat error: {e}")

        elif st.session_state.transcription:
            st.info("🤖 AI Assistant is ready! Please ensure your OpenAI API key is configured.")
        else:
            st.info("📹 Please process a video first to start chatting!")

    def run(self):
        """Main app runner"""
        st.title("🎥 YouTube AI Chatbot")
        st.markdown("Upload a YouTube video, get its transcription, and chat with an AI about the content!")

        # Render sidebar
        api_key = self.render_sidebar()

        if api_key:
            # Render video section
            self.render_video_section(api_key)

            # Render video player if video is processed
            if st.session_state.video_info:
                self.render_video_player()

            # Render chat interface
            self.render_chat_interface()
        else:
            st.warning("⚠️ Please configure your OpenAI API key in the sidebar to get started.")


if __name__ == "__main__":
    app = YouTubeTranscriptionApp()
    app.run()