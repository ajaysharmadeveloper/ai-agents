import streamlit as st
import os
import tempfile
import re
import json
from typing import List, Dict, Any
import logging
from datetime import datetime
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

# Load environment variables
load_dotenv()

# Fix OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Core libraries
import yt_dlp
from faster_whisper import WhisperModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.callbacks.manager import get_openai_callback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenTracker:
    """Class to track and manage token usage and costs"""

    def __init__(self, file_path="token_usage.json"):
        self.file_path = file_path
        self.load_usage()

    def load_usage(self):
        """Load token usage from file"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.total_tokens = data.get('total_tokens', 0)
                    self.total_cost = data.get('total_cost', 0.0)
            else:
                self.total_tokens = 0
                self.total_cost = 0.0
        except Exception as e:
            logger.error(f"Error loading token usage: {e}")
            self.total_tokens = 0
            self.total_cost = 0.0

    def save_usage(self):
        """Save token usage to file"""
        try:
            data = {
                'total_tokens': self.total_tokens,
                'total_cost': self.total_cost,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving token usage: {e}")

    def update_usage(self, tokens: int, cost: float):
        """Update token usage and cost"""
        self.total_tokens += tokens
        self.total_cost += cost
        self.save_usage()

    def get_stats(self):
        """Get current usage statistics"""
        return {
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost
        }


class GoogleSheetsManager:
    """Class to manage Google Sheets operations"""

    def __init__(self):
        self.setup_credentials()

    def setup_credentials(self):
        """Setup Google Sheets credentials"""
        try:
            # Check if all required environment variables are present
            required_vars = [
                "GOOGLE_PROJECT_ID",
                "GOOGLE_PRIVATE_KEY_ID",
                "GOOGLE_PRIVATE_KEY",
                "GOOGLE_CLIENT_EMAIL",
                "GOOGLE_CLIENT_ID",
                "GOOGLE_SPREADSHEET_ID"
            ]

            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)

            if missing_vars:
                logger.error(f"Missing Google Sheets environment variables: {missing_vars}")
                self.gc = None
                return

            # Get credentials from environment variables
            private_key = os.getenv("GOOGLE_PRIVATE_KEY", "").replace('\\n', '\n')
            client_email = os.getenv("GOOGLE_CLIENT_EMAIL", "")

            service_account_info = {
                "type": "service_account",
                "project_id": os.getenv("GOOGLE_PROJECT_ID"),
                "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
                "private_key": private_key,
                "client_email": client_email,
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email}"
            }

            scopes = ['https://www.googleapis.com/auth/spreadsheets']
            credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)
            self.gc = gspread.authorize(credentials)
            self.spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")

            # Test the connection
            try:
                sheet = self.gc.open_by_key(self.spreadsheet_id)
                logger.info("Google Sheets connection successful")
            except Exception as test_error:
                logger.error(f"Failed to access Google Spreadsheet: {test_error}")
                self.gc = None

        except Exception as e:
            logger.error(f"Error setting up Google Sheets credentials: {e}")
            self.gc = None

    def save_user_data(self, name: str, email: str, mobile: str):
        """Save user data to Google Sheets"""
        try:
            if not self.gc:
                return False

            sheet = self.gc.open_by_key(self.spreadsheet_id)

            # Try to get or create users worksheet
            try:
                worksheet = sheet.worksheet("users")
            except:
                worksheet = sheet.add_worksheet(title="users", rows="1000", cols="20")
                # Add headers
                worksheet.append_row(["Timestamp", "Name", "Email", "Mobile"])

            # Add user data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row([timestamp, name, email, mobile])

            return True

        except Exception as e:
            logger.error(f"Error saving user data: {e}")
            return False

    def save_feedback(self, name: str, email: str, feedback: str, rating: int):
        """Save feedback to Google Sheets"""
        try:
            if not self.gc:
                return False

            sheet = self.gc.open_by_key(self.spreadsheet_id)

            # Try to get or create feedback worksheet
            try:
                worksheet = sheet.worksheet("feedback")
            except:
                worksheet = sheet.add_worksheet(title="feedback", rows="1000", cols="20")
                # Add headers
                worksheet.append_row(["Timestamp", "Name", "Email", "Feedback", "Rating"])

            # Add feedback data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row([timestamp, name, email, feedback, rating])

            return True

        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False


class YouTubeTranscriptionApp:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.token_tracker = TokenTracker()
        self.sheets_manager = GoogleSheetsManager()

        # Get OpenAI API key from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="YouTube AI Chatbot",
            page_icon="🎥",
            layout="wide",
            initial_sidebar_state="collapsed"
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
        if 'user_registered' not in st.session_state:
            st.session_state.user_registered = False
        if 'user_name' not in st.session_state:
            st.session_state.user_name = ""
        if 'user_email' not in st.session_state:
            st.session_state.user_email = ""
        if 'user_mobile' not in st.session_state:
            st.session_state.user_mobile = ""

    def render_user_registration_modal(self):
        """Render user registration modal"""
        if not st.session_state.user_registered:
            st.markdown("# 👋 Welcome to YouTube AI Chatbot")
            st.markdown("Please provide your details to get started:")

            with st.form("user_registration"):
                name = st.text_input("Full Name *", placeholder="Enter your full name")
                email = st.text_input("Email Address *", placeholder="Enter your email address")
                mobile = st.text_input("Mobile Number *", placeholder="Enter your mobile number")

                submitted = st.form_submit_button("💾 Save", use_container_width=True, type="primary")

                if submitted:
                    if name and email and mobile:
                        # Simple email validation
                        if "@" in email and "." in email:
                            # Save to Google Sheets
                            try:
                                success = self.sheets_manager.save_user_data(name, email, mobile)

                                if success:
                                    st.session_state.user_registered = True
                                    st.session_state.user_name = name
                                    st.session_state.user_email = email
                                    st.session_state.user_mobile = mobile
                                    st.success("✅ Registration successful! Welcome to the app.")
                                    st.rerun()
                                else:
                                    st.error(
                                        "❌ Failed to save user data. Please check your Google Sheets configuration.")

                            except Exception as e:
                                logger.error(f"Error saving user data: {e}")
                                st.error(f"❌ Failed to save user data. Error: {str(e)}")
                                st.error("Please check your .env file and Google Sheets configuration.")
                        else:
                            st.error("❌ Please enter a valid email address.")
                    else:
                        st.error("❌ Please fill in all required fields.")

    def render_token_stats(self):
        """Render token usage statistics"""
        stats = self.token_tracker.get_stats()

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            st.metric("Total Tokens Used", f"{stats['total_tokens']:,}")

        with col2:
            st.metric("Total Cost", f"${stats['total_cost']:.4f}")

        with col3:
            st.info(f"👋 Welcome back, {st.session_state.user_name}!")

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

    def find_ffmpeg(self):
        """Try to find ffmpeg in various locations"""
        import shutil

        # Try imageio-ffmpeg first (most reliable)
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        except Exception:
            pass

        # Try to find ffmpeg in PATH
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path

        # Common paths where ffmpeg might be installed
        common_paths = [
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
            '/opt/conda/bin/ffmpeg',
            '/home/appuser/venv/bin/ffmpeg',
            '/app/.apt/usr/bin/ffmpeg',  # Streamlit Cloud path
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def download_audio(self, url: str) -> str:
        """Download audio from YouTube video"""
        import shutil

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try to find ffmpeg path
                ffmpeg_path = self.find_ffmpeg()

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

                # Add ffmpeg location if found
                if ffmpeg_path:
                    ydl_opts['ffmpeg_location'] = ffmpeg_path
                else:
                    st.warning("⚠️ FFmpeg not found, will try without conversion")

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                # Find the downloaded file
                for file in os.listdir(temp_dir):
                    if file.endswith(('.mp3', '.m4a', '.webm', '.mp4')):
                        audio_path = os.path.join(temp_dir, file)

                        # Create a new temporary file in the working directory
                        file_ext = file.split('.')[-1]
                        permanent_path = tempfile.NamedTemporaryFile(
                            suffix=f'.{file_ext}',
                            delete=False,
                            dir=os.getcwd()
                        ).name

                        # Copy file instead of moving (fixes cross-device link error)
                        shutil.copy2(audio_path, permanent_path)
                        return permanent_path

        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            st.error(f"Audio download failed: {str(e)}")
            return None

    def download_audio_alternative(self, url: str) -> str:
        """Download audio from YouTube video without FFmpeg conversion"""
        import shutil

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'format': 'worstaudio/worst',  # Use worst quality to avoid conversion
                    'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'extractaudio': False,  # Don't extract audio, download as-is
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                # Find the downloaded file
                for file in os.listdir(temp_dir):
                    if any(file.endswith(ext) for ext in ['.webm', '.m4a', '.mp4', '.mp3']):
                        audio_path = os.path.join(temp_dir, file)

                        # Create a new temporary file in the working directory
                        file_ext = file.split('.')[-1]
                        permanent_path = tempfile.NamedTemporaryFile(
                            suffix=f'.{file_ext}',
                            delete=False,
                            dir=os.getcwd()
                        ).name

                        # Copy file instead of moving
                        shutil.copy2(audio_path, permanent_path)
                        return permanent_path

        except Exception as e:
            logger.error(f"Error downloading audio (alternative): {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using faster-whisper"""
        try:
            # Load model with CPU-optimized settings
            model = WhisperModel("base", device="cpu", compute_type="int8")

            # Transcribe the audio
            segments, info = model.transcribe(
                audio_path,
                beam_size=1,  # Faster inference
                language="en",  # Can be set to None for auto-detection
                condition_on_previous_text=False  # Faster processing
            )

            # Combine all segments into full text
            transcription = " ".join([segment.text for segment in segments])

            return transcription.strip()

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            st.error(f"Transcription failed: {str(e)}")
            return ""
        finally:
            # Clean up temporary audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as cleanup_error:
                    logger.warning(f"Could not cleanup audio file: {cleanup_error}")
                    pass  # Ignore cleanup errors

    def setup_qa_chain(self, transcription: str, video_info: Dict):
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
            embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
            vector_store = FAISS.from_documents(documents, embeddings)

            # Create conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Create the LLM with modern ChatOpenAI
            llm = ChatOpenAI(
                api_key=self.openai_api_key,
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

    def render_video_section(self):
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

                        # If FFmpeg fails, try alternative method
                        if not audio_path:
                            st.warning("⚠️ Standard download failed, trying alternative method...")
                            audio_path = self.download_audio_alternative(video_url)

                        if audio_path:
                            # Transcribe audio
                            st.info("🎤 Transcribing audio...")
                            transcription = self.transcribe_audio(audio_path)

                            if transcription:
                                st.session_state.transcription = transcription

                                # Setup QA chain
                                st.info("🤖 Setting up AI assistant...")
                                qa_chain, vector_store = self.setup_qa_chain(
                                    transcription, video_info
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

            # Use regular text input with button instead of st.chat_input()
            with st.form("chat_form", clear_on_submit=True):
                col1, col2 = st.columns([4, 1])

                with col1:
                    prompt = st.text_input(
                        "Your question:",
                        placeholder="Ask about the video...",
                        label_visibility="collapsed"
                    )

                with col2:
                    submitted = st.form_submit_button("Send", use_container_width=True)

                if submitted and prompt:
                    # Add user message to chat history
                    with st.chat_message("user"):
                        st.write(prompt)

                    # Get AI response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                # Track token usage
                                with get_openai_callback() as cb:
                                    # Use invoke instead of __call__ for modern LangChain
                                    response = st.session_state.qa_chain.invoke({
                                        "question": prompt,
                                        "chat_history": []  # Reset for each question to avoid memory issues
                                    })
                                    answer = response["answer"]
                                    st.write(answer)

                                    # Update token usage
                                    self.token_tracker.update_usage(cb.total_tokens, cb.total_cost)

                                # Add to chat history
                                st.session_state.chat_history.append((prompt, answer))

                                # Rerun to show the new conversation
                                st.rerun()

                            except Exception as e:
                                error_msg = f"Error processing your question: {str(e)}"
                                st.error(error_msg)
                                logger.error(f"Chat error: {e}")

        elif st.session_state.transcription:
            st.info("🤖 AI Assistant is ready!")
        else:
            st.info("📹 Please process a video first to start chatting!")

    def render_feedback_section(self):
        """Render feedback section"""
        st.header("📝 Feedback")
        st.write("Help us improve! Share your experience with the app.")

        with st.form("feedback_form"):
            col1, col2 = st.columns(2)

            with col1:
                rating = st.select_slider(
                    "Rate your experience:",
                    options=[1, 2, 3, 4, 5],
                    value=5,
                    format_func=lambda x: "⭐" * x
                )

            with col2:
                st.write("")  # Empty space for alignment

            feedback_text = st.text_area(
                "Your feedback:",
                placeholder="Tell us what you think about the app, any issues you faced, or suggestions for improvement...",
                height=100
            )

            feedback_submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

            if feedback_submitted:
                if feedback_text.strip():
                    if self.sheets_manager.save_feedback(
                            st.session_state.user_name,
                            st.session_state.user_email,
                            feedback_text,
                            rating
                    ):
                        st.success("Thank you for your feedback! 🎉")
                    else:
                        st.error("Failed to save feedback. Please try again.")
                else:
                    st.error("Please provide your feedback before submitting.")

    def run(self):
        """Main app runner"""
        # Check if user is registered
        if not st.session_state.user_registered:
            self.render_user_registration_modal()
            return

        # Check if OpenAI API key is available
        if not self.openai_api_key:
            st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            return

        # Main app content
        st.title("🎥 YouTube AI Chatbot")
        st.markdown("Upload a YouTube video, get its transcription, and chat with an AI about the content!")

        # Render token usage statistics
        self.render_token_stats()

        st.divider()

        # Render video section
        self.render_video_section()

        # Render video player if video is processed
        if st.session_state.video_info:
            self.render_video_player()

        # Render chat interface
        self.render_chat_interface()

        # Render feedback section after chat interface
        if st.session_state.qa_chain and st.session_state.transcription:
            self.render_feedback_section()


if __name__ == "__main__":
    app = YouTubeTranscriptionApp()
    app.run()