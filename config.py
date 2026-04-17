"""
Configuration file for Sri Lanka Sentiment Analyzer
Contains all project settings and constants
"""
import os

# ============================================================================
# PROJECT ROOT (FIXED FOR STREAMLIT CLOUD)
# ============================================================================
# config.py is at project root, so BASE_DIR = its own directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist (safe for Streamlit + local)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================
RAW_TWEETS_FILE = 'tweets.csv'
PROCESSED_TWEETS_FILE = 'processed_tweets.csv'
LABELED_TWEETS_FILE = 'labeled_tweets.csv'

# Model files
MODEL_FILE = 'sentiment_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'

# ============================================================================
# POLICY KEYWORDS
# ============================================================================
POLICY_KEYWORDS = [
    'government', 'policy', 'tax', 'budget', 'law', 'protest',
    'crisis', 'economy', 'inflation', 'minister', 'parliament',
    'election', 'vote', 'president', 'reform', 'corruption',
    'fuel', 'power cut', 'subsidy', 'healthcare', 'education'
]

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
STOPWORDS_LANGUAGE = 'english'
CUSTOM_STOPWORDS = [
    'rt', 'http', 'https', 'www', 'com', 'co', 'amp',
    'sri', 'lanka', 'lka', 'srilanka', 'srilankan'
]

# ============================================================================
# MODEL SETTINGS
# ============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.8

# ============================================================================
# SENTIMENT LABELS
# ============================================================================
SENTIMENT_LABELS = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

SENTIMENT_COLORS = {
    'Negative': '#FF4444',
    'Neutral': '#FFB800',
    'Positive': '#00C851'
}

# ============================================================================
# VISUALIZATION
# ============================================================================
FIGURE_SIZE = (12, 6)
DPI = 100
PRIMARY_COLOR = '#1E88E5'
SECONDARY_COLOR = '#FFC107'

# ============================================================================
# STREAMLIT SETTINGS
# ============================================================================
PAGE_TITLE = "🇱🇰 Sri Lanka Public Policy Sentiment Analyzer"
PAGE_ICON = "📊"
LAYOUT = "wide"
SAMPLE_TWEETS_COUNT = 10

# ============================================================================
# TRAINING DATA
# ============================================================================
TRAINING_DATA_URL = None
