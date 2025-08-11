import streamlit as st
import os
import tempfile
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
import time
import torch

# Load environment variables
load_dotenv()

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
from langchain_community.callbacks.manager import get_openai_callback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenTracker:
    """Track OpenAI API token usage and costs"""

    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.session_tokens = 0
        self.session_cost = 0.0

        # OpenAI pricing (as of 2024)
        self.pricing = {
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},  # per 1K tokens
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'text-embedding-ada-002': {'input': 0.0001, 'output': 0.0001}
        }

    def update_usage(self, tokens_used: int, model: str = 'gpt-3.5-turbo', token_type: str = 'input'):
        """Update token usage and calculate costs"""
        self.total_tokens += tokens_used
        self.session_tokens += tokens_used

        cost_per_1k = self.pricing.get(model, {}).get(token_type, 0.002)
        cost = (tokens_used / 1000) * cost_per_1k

        self.total_cost += cost
        self.session_cost += cost

        return cost

    def get_stats(self) -> Dict:
        """Get current usage statistics"""
        return {
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'session_tokens': self.session_tokens,
            'session_cost': self.session_cost
        }

    def reset_session(self):
        """Reset session counters"""
        self.session_tokens = 0
        self.session_cost = 0.0


class GoogleSheetsManager:
    """Manage Google Sheets integration for user data and feedback"""

    def __init__(self):
        self.gc = None
        self.spreadsheet = None
        self.setup_sheets_connection()

    def setup_sheets_connection(self):
        """Setup Google Sheets connection using service account"""
        try:
            # Load service account credentials from environment
            service_account_info = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            if service_account_info:
                credentials_dict = json.loads(service_account_info)
                credentials = Credentials.from_service_account_info(
                    credentials_dict,
                    scopes=['https://www.googleapis.com/auth/spreadsheets']
                )
                self.gc = gspread.authorize(credentials)

                # Open or create spreadsheet
                spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")
                if spreadsheet_id:
                    self.spreadsheet = self.gc.open_by_key(spreadsheet_id)
                    logger.info("Google Sheets connection established")
            else:
                logger.warning("Google Sheets credentials not found")
        except Exception as e:
            logger.error(f"Failed to setup Google Sheets: {e}")
            self.gc = None

    def save_user_data(self, user_data: Dict):
        """Save user registration data"""
        if not self.spreadsheet:
            return False

        try:
            worksheet = self.spreadsheet.worksheet("Users")
            worksheet.append_row([
                datetime.now().isoformat(),
                user_data.get('name', ''),
                user_data.get('email', ''),
                user_data.get('mobile', '')
            ])
            return True
        except Exception as e:
            logger.error(f"Failed to save user data: {e}")
            return False

    def save_feedback(self, feedback_data: Dict):
        """Save user feedback"""
        if not self.spreadsheet:
            return False

        try:
            worksheet = self.spreadsheet.worksheet("Feedback")
            worksheet.append_row([
                datetime.now().isoformat(),
                feedback_data.get('user_name', ''),
                feedback_data.get('video_title', ''),
                feedback_data.get('rating', ''),
                feedback_data.get('feedback', ''),
                feedback_data.get('language', '')
            ])
            return True
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False


class LanguageDetector:
    """Intelligent language detection for videos"""

    @staticmethod
    def detect_language_from_metadata(video_info: Dict) -> Tuple[Optional[str], float]:
        """Detect language from video metadata with confidence score"""
        try:
            title = video_info.get('title', '').lower()
            description = video_info.get('description', '').lower()
            uploader = video_info.get('uploader', '').lower()
            tags = [tag.lower() for tag in video_info.get('tags', [])]

            # Combine all text
            all_text = f"{title} {description} {uploader} {' '.join(tags)}"

            # Hindi indicators with weights
            hindi_indicators = {
                # Strong indicators (high weight)
                'हिंदी': 3.0, 'हिन्दी': 3.0, 'देवनागरी': 3.0,
                'भारत': 2.5, 'इंडिया': 2.5, 'नमस्ते': 2.5,
                'धन्यवाद': 2.0, 'जी हां': 2.0, 'क्या': 2.0,

                # Medium indicators
                'hindi': 2.0, 'bollywood': 1.5, 'bharat': 1.5,
                'desi': 1.5, 'india': 1.0, 'indian': 1.0,

                # Context indicators
                'film': 0.5, 'movie': 0.5, 'song': 0.5,
                'music': 0.3, 'dance': 0.3
            }

            # English indicators
            english_indicators = {
                'english': 2.0, 'tutorial': 1.5, 'how to': 1.5,
                'review': 1.0, 'tech': 1.0, 'programming': 1.5,
                'coding': 1.5, 'learn': 1.0, 'education': 1.0
            }

            # Calculate scores
            hindi_score = 0.0
            english_score = 0.0

            for indicator, weight in hindi_indicators.items():
                if indicator in all_text:
                    hindi_score += weight

            for indicator, weight in english_indicators.items():
                if indicator in all_text:
                    english_score += weight

            # Check for Devanagari script
            devanagari_chars = len(re.findall(r'[\u0900-\u097F]', all_text))
            if devanagari_chars > 0:
                hindi_score += devanagari_chars * 0.5

            # Determine language
            total_score = hindi_score + english_score
            if total_score == 0:
                return None, 0.0

            if hindi_score > english_score:
                confidence = min(hindi_score / (total_score + 1), 0.95)
                return 'hi', confidence
            elif english_score > hindi_score:
                confidence = min(english_score / (total_score + 1), 0.95)
                return 'en', confidence
            else:
                return None, 0.0

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return None, 0.0


class OptimizedWhisperTranscriber:
    """Optimized transcription handler using OpenAI Whisper"""

    def __init__(self):
        self._models = {}  # Cache for different models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.supported_languages = {
            'en': {
                'name': 'English',
                'flag': '🇺🇸',
                'models': ['tiny.en', 'base.en', 'small.en', 'medium.en'],
                'default_model': 'base.en',
                'speed_multiplier': 1.0
            },
            'hi': {
                'name': 'Hindi',
                'flag': '🇮🇳',
                'models': ['base', 'small', 'medium', 'large-v3'],
                'default_model': 'large-v3',  # Best for Hindi
                'speed_multiplier': 2.5
            }
        }

    def get_model(self, model_name: str) -> whisper.Whisper:
        """Get cached model or create new one"""
        if model_name not in self._models:
            logger.info(f"Loading Whisper model: {model_name}")
            with st.spinner(f"Loading {model_name} model..."):
                self._models[model_name] = whisper.load_model(
                    model_name,
                    device=self.device,
                    download_root="./whisper_models"
                )
            logger.info(f"Model {model_name} loaded successfully on {self.device}")

        return self._models[model_name]

    def get_optimal_settings(self, language: str, model_name: str, duration: float) -> Dict:
        """Get optimal transcription settings based on language and video duration"""

        base_settings = {
            'verbose': True,
            'temperature': 0.0,
            'compression_ratio_threshold': 2.4,
            'logprob_threshold': -1.0,
            'no_speech_threshold': 0.6,
            'condition_on_previous_text': True,
            'fp16': self.device == "cuda",
        }

        if language == 'hi':
            # Hindi-specific optimizations
            settings = {
                **base_settings,
                'language': 'hi',
                'task': 'transcribe',
                'beam_size': 5,  # Better for Hindi
                'best_of': 5,
                'patience': 1.0,
                'length_penalty': 1.0,
                'temperature': 0.0,
                'initial_prompt': "यह एक हिंदी वीडियो है।",
                'suppress_tokens': "",
                'word_timestamps': False,  # Faster processing
            }
        else:
            # English/Default optimizations
            settings = {
                **base_settings,
                'language': 'en' if language == 'en' else None,
                'task': 'transcribe',
                'beam_size': 5,
                'best_of': 5,
                'patience': 1.0,
                'word_timestamps': False,
            }

        return settings

    def estimate_transcription_time(self, duration: float, language: str, model: str) -> Tuple[float, float]:
        """Estimate transcription time range"""
        lang_info = self.supported_languages.get(language, self.supported_languages['en'])

        # Base time calculation (empirical) - adjusted for Whisper
        model_multipliers = {
            'tiny': 0.03,
            'tiny.en': 0.03,
            'base': 0.05,
            'base.en': 0.05,
            'small': 0.1,
            'small.en': 0.1,
            'medium': 0.2,
            'medium.en': 0.2,
            'large': 0.3,
            'large-v2': 0.3,
            'large-v3': 0.35
        }

        base_time = duration * model_multipliers.get(model, 0.2)
        language_multiplier = lang_info['speed_multiplier']

        # GPU acceleration factor
        gpu_factor = 0.3 if self.device == "cuda" else 1.0

        min_time = base_time * language_multiplier * gpu_factor * 0.8
        max_time = base_time * language_multiplier * gpu_factor * 1.5

        return min_time, max_time

    def transcribe_audio(self, audio_path: str, language: str, model_name: str,
                         progress_callback=None) -> Tuple[str, Dict]:
        """Transcribe audio using OpenAI Whisper"""
        try:
            # Load model
            model = self.get_model(model_name)

            # Get optimal settings
            duration = self.get_audio_duration(audio_path)
            settings = self.get_optimal_settings(language, model_name, duration)

            # Transcribe
            logger.info(f"Starting transcription with {model_name} for {language}")
            start_time = time.time()

            result = model.transcribe(audio_path, **settings)

            end_time = time.time()
            processing_time = end_time - start_time

            # Extract transcription and metadata
            transcription = result['text'].strip()

            info = {
                'language': result.get('language', language),
                'duration': processing_time,
                'segments': len(result.get('segments', [])),
                'model': model_name,
                'device': self.device
            }

            logger.info(f"Transcription completed in {processing_time:.1f}s")

            return transcription, info

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using whisper's audio loading"""
        try:
            audio = whisper.load_audio(audio_path)
            duration = len(audio) / 16000  # Whisper uses 16kHz sampling
            return duration
        except:
            return 0.0


class YouTubeTranscriptionApp:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.token_tracker = TokenTracker()
        self.sheets_manager = GoogleSheetsManager()
        self.language_detector = LanguageDetector()
        self.transcriber = OptimizedWhisperTranscriber()

        # Get OpenAI API key from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="YouTube AI Chatbot - OpenAI Whisper",
            page_icon="🎥",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

    def initialize_session_state(self):
        """Initialize session state variables"""
        session_vars = [
            'transcription', 'chat_history', 'video_info', 'qa_chain',
            'vector_store', 'user_registered', 'user_name', 'user_email',
            'user_mobile', 'detected_language', 'language_confidence'
        ]

        for var in session_vars:
            if var not in st.session_state:
                if var in ['chat_history']:
                    st.session_state[var] = []
                elif var in ['video_info']:
                    st.session_state[var] = {}
                elif var in ['user_registered']:
                    st.session_state[var] = False
                else:
                    st.session_state[var] = ""

    def render_user_registration_modal(self):
        """Render user registration form"""
        st.title("🎥 YouTube AI Chatbot with OpenAI Whisper")
        st.markdown("**Welcome! Please register to continue**")

        with st.form("user_registration"):
            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input("Name *", placeholder="Enter your full name")
                email = st.text_input("Email *", placeholder="your@email.com")

            with col2:
                mobile = st.text_input("Mobile", placeholder="+91 XXXXXXXXXX")

            submitted = st.form_submit_button("🚀 Start Using App", type="primary")

            if submitted:
                if name and email:
                    # Save user data
                    user_data = {
                        'name': name,
                        'email': email,
                        'mobile': mobile
                    }

                    # Update session state
                    st.session_state.user_registered = True
                    st.session_state.user_name = name
                    st.session_state.user_email = email
                    st.session_state.user_mobile = mobile

                    # Save to Google Sheets
                    self.sheets_manager.save_user_data(user_data)

                    st.rerun()
                else:
                    st.error("Please fill in required fields (Name and Email)")

    def render_token_stats(self):
        """Render token usage statistics"""
        stats = self.token_tracker.get_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Session Tokens", f"{stats['session_tokens']:,}")
        with col2:
            st.metric("Session Cost", f"${stats['session_cost']:.4f}")
        with col3:
            st.metric("Total Tokens", f"{stats['total_tokens']:,}")
        with col4:
            st.metric("Total Cost", f"${stats['total_cost']:.4f}")

    def validate_youtube_url(self, url: str) -> bool:
        """Validate YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)',
            r'(?:youtube\.com/embed/)([^&\n?#]+)',
        ]
        return any(re.search(pattern, url) for pattern in patterns)

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)',
            r'(?:youtube\.com/embed/)([^&\n?#]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return ""

    def extract_video_info(self, url: str) -> Dict:
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
                    'video_id': info.get('id', ''),
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'uploader': info.get('uploader', ''),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'tags': info.get('tags', []),
                    'thumbnail': info.get('thumbnail', ''),
                    'upload_date': info.get('upload_date', ''),
                }
        except Exception as e:
            logger.error(f"Failed to extract video info: {e}")
            return {}

    def download_audio_optimized(self, url: str) -> Optional[str]:
        """Download audio from YouTube robustly, avoiding 403s."""
        try:
            output_dir = tempfile.mkdtemp()
            # Save directly as .m4a to avoid postprocessing + fewer failures
            outtmpl = os.path.join(output_dir, "audio.%(ext)s")

            common_headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/126.0.0.0 Safari/537.36"),
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.youtube.com/"
            }

            # First attempt: Android client (often bypasses throttling)
            primary_opts = {
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
                "retries": 10,
                "fragment_retries": 10,
                "concurrent_fragment_downloads": 3,
                "noprogress": True,
                "http_headers": common_headers,
                "geo_bypass": True,
                "nocheckcertificate": True,
                "extractor_args": {
                    "youtube": {
                        # Try Android first; fallback will try web
                        "player_client": ["android"],
                        # Grab DASH/HLS manifests if needed
                        "include_live_dash": ["true"]
                    }
                },
                # Force IPv4 in some hosts where v6 causes 403s
                "force_ip": "0.0.0.0",
            }

            # Fallback: Web client + explicit m4a preference
            fallback_opts = {
                **primary_opts,
                "extractor_args": {"youtube": {"player_client": ["web"]}},
            }

            def _try_download(opts) -> Optional[str]:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
                # Return the first audio file we see
                for fname in os.listdir(output_dir):
                    if fname.lower().endswith((".m4a", ".mp3", ".wav")):
                        return os.path.join(output_dir, fname)
                return None

            # Try primary, then fallback
            path = _try_download(primary_opts)
            if not path:
                path = _try_download(fallback_opts)

            return path
        except Exception as e:
            logger.error(f"Audio download failed: {e}")
            return None

    def download_audio_alternative(self, url: str) -> Optional[str]:
        """Alternative audio download method"""
        try:
            output_dir = tempfile.mkdtemp()
            output_path = os.path.join(output_dir, "audio.m4a")

            ydl_opts = {
                'format': 'worstaudio',
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            return output_path if os.path.exists(output_path) else None
        except Exception as e:
            logger.error(f"Alternative audio download failed: {e}")
            return None

    def post_process_hindi_text(self, text: str) -> str:
        """Post-process Hindi transcription"""
        # Basic cleaning for Hindi text
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = text.strip()

        # Remove any artifacts or repeated characters
        text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)  # Limit repeated chars to 3

        return text

    def setup_qa_chain(self, transcription: str, video_info: Dict) -> Tuple[Optional[Any], Optional[Any]]:
        """Setup QA chain with video transcription"""
        try:
            # Create documents
            documents = [Document(
                page_content=transcription,
                metadata={
                    'title': video_info.get('title', ''),
                    'uploader': video_info.get('uploader', ''),
                    'duration': video_info.get('duration', 0),
                    'video_id': video_info.get('video_id', '')
                }
            )]

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            splits = text_splitter.split_documents(documents)

            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(splits, embeddings)

            # Setup LLM and memory
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Create QA chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                return_source_documents=True,
                verbose=True
            )

            return qa_chain, vector_store

        except Exception as e:
            logger.error(f"Failed to setup QA chain: {e}")
            return None, None

    def render_smart_model_selection(self):
        """Smart model selection based on detected language"""
        st.subheader("⚙️ Transcription Settings")

        # Get detected language info
        detected_lang = st.session_state.get('detected_language')
        confidence = st.session_state.get('language_confidence', 0.0)

        col1, col2, col3 = st.columns(3)

        with col1:
            # Language override option
            language_options = ["Auto-detect", "Force English 🇺🇸", "Force Hindi 🇮🇳"]

            if detected_lang:
                lang_info = self.transcriber.supported_languages.get(detected_lang, {})
                flag = lang_info.get('flag', '🌍')
                name = lang_info.get('name', 'Unknown')
                default_idx = 1 if detected_lang == 'en' else 2
                st.info(f"**Detected:** {flag} {name} ({confidence:.1%} confidence)")
            else:
                default_idx = 0

            language_mode = st.selectbox(
                "Language Mode:",
                options=language_options,
                index=default_idx,
                help="Auto-detect uses video metadata analysis"
            )

        with col2:
            # Model selection based on language
            if "Hindi" in language_mode or (detected_lang == 'hi' and "Auto" in language_mode):
                available_models = self.transcriber.supported_languages['hi']['models']
                default_model = self.transcriber.supported_languages['hi']['default_model']
                default_idx = available_models.index(default_model)
                st.info("💡 Using large-v3 for best Hindi accuracy")
            else:
                available_models = self.transcriber.supported_languages['en']['models']
                default_model = self.transcriber.supported_languages['en']['default_model']
                default_idx = available_models.index(default_model)

            model_choice = st.selectbox(
                "Model Quality:",
                options=available_models,
                index=default_idx,
                help="large-v3 provides best accuracy for Hindi"
            )

        with col3:
            # Speed estimation
            duration = st.session_state.video_info.get('duration', 0)
            if duration > 0:
                final_lang = self.get_final_language(language_mode, detected_lang)
                min_time, max_time = self.transcriber.estimate_transcription_time(
                    duration, final_lang, model_choice
                )

                if min_time < 60:
                    time_str = f"{min_time:.0f}-{max_time:.0f}s"
                else:
                    time_str = f"{min_time / 60:.1f}-{max_time / 60:.1f}m"

                st.metric("Est. Time", time_str)

                # Show GPU status
                if self.transcriber.device == "cuda":
                    st.success("🚀 GPU Acceleration Active")
                else:
                    st.warning("🐌 CPU Mode (Slower)")

        return language_mode, model_choice

    def get_final_language(self, language_mode: str, detected_lang: str) -> str:
        """Determine final language based on mode and detection"""
        if "English" in language_mode:
            return 'en'
        elif "Hindi" in language_mode:
            return 'hi'
        else:
            return detected_lang or 'en'

    def get_speed_rating(self, model: str, language_mode: str) -> str:
        """Get speed rating for model/language combination"""
        ratings = {
            ('tiny.en', 'English'): '⚡ Lightning Fast',
            ('base.en', 'English'): '🚀 Very Fast',
            ('small.en', 'English'): '🏃 Fast',
            ('medium.en', 'English'): '🚶 Medium',
            ('base', 'Hindi'): '🚶 Medium',
            ('small', 'Hindi'): '🐌 Slow',
            ('medium', 'Hindi'): '🐢 Very Slow',
            ('large-v3', 'Hindi'): '🐌 Slow (High Quality)'
        }

        lang_key = 'Hindi' if 'Hindi' in language_mode else 'English'
        return ratings.get((model, lang_key), '🤔 Variable')

    def analyze_video_intelligence(self, url: str) -> Dict:
        """Intelligent video analysis with language detection"""
        try:
            # Extract video metadata
            video_info = self.extract_video_info(url)
            if not video_info:
                return {}

            # Detect language from metadata
            detected_lang, confidence = self.language_detector.detect_language_from_metadata(video_info)

            # Store in session state
            st.session_state.detected_language = detected_lang
            st.session_state.language_confidence = confidence
            st.session_state.video_info = video_info

            return {
                'video_info': video_info,
                'detected_language': detected_lang,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return {}

    def transcribe_with_intelligent_optimization(self, audio_path: str, language_mode: str, model_choice: str) -> str:
        """Transcribe with intelligent optimization based on language"""
        try:
            progress_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Determine final language and settings
                detected_lang = st.session_state.get('detected_language')
                final_language = self.get_final_language(language_mode, detected_lang)

                # Show language-specific status
                lang_info = self.transcriber.supported_languages.get(final_language, {})
                flag = lang_info.get('flag', '🌍')
                name = lang_info.get('name', 'Auto')

                status_text.text(f"{flag} Loading {name} optimized model ({model_choice})...")
                progress_bar.progress(20)

                # Get model
                model = self.transcriber.get_model(model_choice)
                progress_bar.progress(40)

                # Transcribe
                status_text.text(f"🎤 Transcribing {name} audio with Whisper {model_choice}...")
                progress_bar.progress(50)

                # Transcribe with Whisper
                transcription, info = self.transcriber.transcribe_audio(
                    audio_path, final_language, model_choice
                )

                progress_bar.progress(90)
                status_text.text("📝 Processing transcription...")

                # Language-specific post-processing
                if final_language == 'hi':
                    transcription = self.post_process_hindi_text(transcription)

                progress_bar.progress(100)
                status_text.text("✅ Transcription complete!")

                # Show completion stats
                actual_lang = info.get('language', final_language)
                duration = info.get('duration', 0)

                st.success(
                    f"{flag} {name} transcription completed in {duration:.1f}s using {info.get('device', 'CPU')}")
                st.info(f"📊 Model: {model_choice} | Language: {actual_lang} | Segments: {info.get('segments', 0)}")

                # Clear progress
                time.sleep(1)
                progress_container.empty()

                return transcription.strip()

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            st.error(f"❌ Transcription failed: {str(e)}")
            return ""

    def render_video_preview(self):
        """Render video preview with language info"""
        info = st.session_state.video_info
        detected_lang = st.session_state.get('detected_language')

        with st.expander("📺 Video Information", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Title:** {info.get('title', 'Unknown')}")
                st.write(f"**Channel:** {info.get('uploader', 'Unknown')}")
                duration_min = info.get('duration', 0) / 60
                st.write(f"**Duration:** {duration_min:.1f} minutes")

            with col2:
                st.write(f"**Views:** {info.get('view_count', 0):,}")
                if detected_lang:
                    lang_info = self.transcriber.supported_languages.get(detected_lang, {})
                    st.write(f"**Detected Language:** {lang_info.get('flag', '')} {lang_info.get('name', 'Unknown')}")

    def render_intelligent_video_section(self):
        """Enhanced video section with intelligent language handling"""
        st.header("🎥 YouTube AI Chatbot")
        st.markdown("**Powered by OpenAI Whisper - Best-in-class Hindi & English transcription**")

        # Show GPU/CPU status
        device_status = "🚀 GPU Acceleration Available" if self.transcriber.device == "cuda" else "💻 Running on CPU"
        st.info(device_status)

        # Video URL input
        video_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL (English or Hindi supported)"
        )

        if video_url and self.validate_youtube_url(video_url):
            # Analyze video when URL is entered
            if st.session_state.video_info.get('video_id') != self.extract_video_id(video_url):
                with st.spinner("🔍 Analyzing video..."):
                    analysis = self.analyze_video_intelligence(video_url)

                    if analysis:
                        detected_lang = analysis.get('detected_language')
                        confidence = analysis.get('confidence', 0)

                        if detected_lang:
                            lang_info = self.transcriber.supported_languages[detected_lang]
                            st.success(
                                f"🎯 Detected: {lang_info['flag']} {lang_info['name']} content ({confidence:.1%} confidence)")
                        else:
                            st.info("🤔 Language auto-detection will be used during processing")

            # Show video info and model selection
            if st.session_state.video_info:
                self.render_video_preview()

                # Smart model selection
                language_mode, model_choice = self.render_smart_model_selection()

                # Duration warnings
                self.show_duration_warnings(language_mode)

                # Process button
                col1, col2 = st.columns([3, 1])
                with col2:
                    process_button = st.button("🚀 Process Video", type="primary")

                if process_button:
                    self.process_video_intelligently(video_url, language_mode, model_choice)

        elif video_url and not self.validate_youtube_url(video_url):
            st.error("❌ Please enter a valid YouTube URL")

    def show_duration_warnings(self, language_mode: str):
        """Show intelligent duration warnings based on language"""
        duration = st.session_state.video_info.get('duration', 0)
        duration_minutes = duration / 60

        if duration_minutes > 0:
            if "Hindi" in language_mode or st.session_state.get('detected_language') == 'hi':
                if duration_minutes > 30:
                    st.warning(f"⚠️ {duration_minutes:.1f}-minute Hindi video will take significant time to process")
                elif duration_minutes > 10:
                    st.info(f"📝 {duration_minutes:.1f}-minute Hindi video - using large-v3 model for best accuracy")
            else:
                if duration_minutes > 60:
                    st.warning(
                        f"⚠️ {duration_minutes:.1f}-minute video - consider using a shorter clip for faster results")
                elif duration_minutes > 20:
                    st.info(f"📝 {duration_minutes:.1f}-minute video - processing will take a few minutes")

    def process_video_intelligently(self, video_url: str, language_mode: str, model_choice: str):
        """Process video with intelligent optimizations"""
        start_time = time.time()

        with st.spinner("Processing video with OpenAI Whisper..."):
            # Download audio
            download_start = time.time()
            st.info("🎵 Downloading audio...")
            audio_path = self.download_audio_optimized(video_url)
            download_time = time.time() - download_start

            if not audio_path:
                st.warning("⚠️ Trying alternative download...")
                audio_path = self.download_audio_alternative(video_url)

            if audio_path:
                st.success(f"✅ Audio downloaded in {download_time:.1f}s")

                # Intelligent transcription
                transcribe_start = time.time()
                transcription = self.transcribe_with_intelligent_optimization(
                    audio_path, language_mode, model_choice
                )
                transcribe_time = time.time() - transcribe_start

                if transcription:
                    st.session_state.transcription = transcription

                    # Setup QA chain
                    st.info("🤖 Setting up AI assistant...")
                    setup_start = time.time()
                    qa_chain, vector_store = self.setup_qa_chain(
                        transcription, st.session_state.video_info
                    )
                    setup_time = time.time() - setup_start

                    if qa_chain:
                        st.session_state.qa_chain = qa_chain
                        st.session_state.vector_store = vector_store

                        total_time = time.time() - start_time

                        # Performance metrics
                        self.show_performance_metrics(
                            download_time, transcribe_time, setup_time, total_time
                        )

                        st.success("✅ Video processed successfully! You can now chat about the content.")
                    else:
                        st.error("❌ Failed to setup AI assistant")
                else:
                    st.error("❌ Transcription failed")
            else:
                st.error("❌ Audio download failed")

    def show_performance_metrics(self, download_time: float, transcribe_time: float,
                                 setup_time: float, total_time: float):
        """Show detailed performance metrics"""
        st.subheader("⚡ Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Download", f"{download_time:.1f}s")
        with col2:
            st.metric("Transcription", f"{transcribe_time:.1f}s")
        with col3:
            st.metric("Setup", f"{setup_time:.1f}s")
        with col4:
            st.metric("Total Time", f"{total_time:.1f}s")

        # Show transcription quality
        if st.session_state.transcription:
            duration = st.session_state.video_info.get('duration', 0)
            word_count = len(st.session_state.transcription.split())
            chars_per_second = len(st.session_state.transcription) / duration if duration > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", f"{word_count:,}")
            with col2:
                st.metric("Chars/Second", f"{chars_per_second:.1f}")
            with col3:
                processing_ratio = transcribe_time / duration if duration > 0 else 0
                st.metric("Speed Ratio", f"{processing_ratio:.2f}x")

    def render_chat_interface(self):
        """Render chat interface for Q&A"""
        if st.session_state.qa_chain and st.session_state.transcription:
            st.header("💬 Chat with Video Content")

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask questions about the video..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.write(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            with get_openai_callback() as cb:
                                response = st.session_state.qa_chain({
                                    "question": prompt,
                                    "chat_history": []
                                })

                                # Track token usage
                                self.token_tracker.update_usage(cb.total_tokens)

                                answer = response.get("answer", "I couldn't find an answer to that question.")
                                st.write(answer)

                                # Add assistant response to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": answer
                                })

                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")

    def render_feedback_section(self):
        """Render feedback collection section"""
        st.header("📝 Feedback")

        with st.expander("Leave Feedback", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                rating = st.selectbox(
                    "How would you rate this transcription?",
                    ["⭐ 1 - Poor", "⭐⭐ 2 - Fair", "⭐⭐⭐ 3 - Good",
                     "⭐⭐⭐⭐ 4 - Very Good", "⭐⭐⭐⭐⭐ 5 - Excellent"]
                )

            with col2:
                language_feedback = st.selectbox(
                    "Language detection accuracy:",
                    ["Perfect", "Good", "Needs improvement", "Completely wrong"]
                )

            feedback_text = st.text_area(
                "Additional comments:",
                placeholder="Share your experience, suggestions, or report issues..."
            )

            if st.button("Submit Feedback"):
                feedback_data = {
                    'user_name': st.session_state.user_name,
                    'video_title': st.session_state.video_info.get('title', ''),
                    'rating': rating,
                    'language_feedback': language_feedback,
                    'feedback': feedback_text,
                    'detected_language': st.session_state.get('detected_language', ''),
                    'language_confidence': st.session_state.get('language_confidence', 0)
                }

                if self.sheets_manager.save_feedback(feedback_data):
                    st.success("Thank you for your feedback!")
                else:
                    st.info("Feedback recorded locally. Thank you!")

    def run(self):
        """Main app runner with bilingual support"""
        # Check user registration
        if not st.session_state.user_registered:
            self.render_user_registration_modal()
            return

        # Check API key
        if not self.openai_api_key:
            st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            return

        # Render token stats
        self.render_token_stats()
        st.divider()

        # Main intelligent video section
        self.render_intelligent_video_section()

        # Render transcription view and chat interface
        if st.session_state.video_info and st.session_state.transcription:
            # Video player (if you want to embed)
            video_id = st.session_state.video_info.get('video_id')
            if video_id:
                st.subheader("🎬 Video Player")
                st.video(f"https://www.youtube.com/watch?v={video_id}")

            # Transcription viewer
            with st.expander("📝 View Transcription", expanded=False):
                st.text_area(
                    "Full Transcription:",
                    st.session_state.transcription,
                    height=300,
                    disabled=True
                )

        # Chat interface
        self.render_chat_interface()

        # Feedback section
        if st.session_state.qa_chain and st.session_state.transcription:
            self.render_feedback_section()


if __name__ == "__main__":
    app = YouTubeTranscriptionApp()
    app.run()