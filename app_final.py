import os
import streamlit as st
import google.generativeai as genai
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import traceback

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- Page config ---
st.set_page_config(page_title="MoodForge: Emotion Aware StoryTelling with NLP", page_icon="📖")

# --- Utilities for dynamic theming ---
THEMES = {
    "positive": {
        "bg": "linear-gradient(135deg, #0f5132 0%, #14532d 100%)",
        "text": "#e6ffee",
        "accent": "#00c853"
    },
    "negative": {
        "bg": "linear-gradient(135deg, #2c0b0e 0%, #58151c 100%)",
        "text": "#ffdddd",
        "accent": "#ff4d4d"
    },
    "neutral": {
        "bg": "linear-gradient(135deg, #212529 0%, #343a40 100%)",
        "text": "#f8f9fa",
        "accent": "#adb5bd"
    }
}

def inject_theme(sentiment: str):
    theme = THEMES.get(sentiment, THEMES["neutral"])
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {theme['bg']} !important;
        }}
        .themed-text, .themed-text * {{
            color: {theme['text']} !important;
        }}
        .themed-accent {{
            border-left: .25rem solid {theme['accent']};
            padding-left: .75rem;
            margin: .25rem 0 .75rem 0;
        }}
        .themed-badge {{
            display: inline-block;
            padding: .25rem .5rem;
            border-radius: 999px;
            background: {theme['accent']}22;
            color: {theme['accent']};
            border: 1px solid {theme['accent']}44;
            font-weight: 600;
            font-size: 0.85rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def sentiment_badge(label: str):
    st.markdown(f"<span class='themed-badge'>{label.title()}</span>", unsafe_allow_html=True)

# --- Load and cache the sentiment model ---
@st.cache_resource
def load_sentiment_model():
    """Loads the sentiment model and caches it."""
    try:
        return joblib.load("sentiment_model1.pkl")
    except FileNotFoundError:
        return None

# --- Preprocess text and cache the result ---
@st.cache_data(show_spinner=False)
def preprocess_text(text):
    """Preprocesses text by cleaning, tokenizing, and lemmatizing."""
    # Stop words
    sentiment_words = {"not", "no", "nor", "never", "don't", "didn't", "isn't", "wasn't", "won't", "can't", "couldn't",
                       "pathetic", "hate", "hates", "hated", "hating", "love", "loves", "loved", "loving", "sad", "sadly",
                       "sadness", "angry", "anger", "angrily", "disgust", "disgusted", "disgusting", "dislikes", "dislike"}
    default_stopwords = set(stopwords.words('english'))
    custom_stopwords = default_stopwords - sentiment_words

    text = re.sub(r'https?:\/\/\S+|www\.\S+|@\w+|<@!?[\d]+>', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in custom_stopwords]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- Generate story (without caching) ---
def generate_story(prediction, word_limit):
    """Generates a story based on sentiment using the Gemini API."""
    prompt = f"Create a unique and interesting short story with normal english words (less than {word_limit} words) based on {prediction} mood."
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the story: {e}"

# --- Main app logic starts here ---
# Hardcode the API key for simplicity as requested.
# Please replace 'YOUR_API_KEY_HERE' with your actual key.
genai.configure(api_key="AIzaSyAT4Tmn5UTjDt92kbPOPAKf_L5iINbyrNk")

# Load the cached model
pipeline = load_sentiment_model()

# --- Main app layout ---
st.title("📖 MoodTales: Emotion Aware StoryTelling with NLP")
st.caption("Turn raw feeling into fiction — powered by your sentiment model + Google Gemini API.")

# Input area
user_input = st.text_area("What's up mate? How are you feeling!", height=150, placeholder="Type or paste some text…")
word_limit = st.slider("Adjust story length (in words)", min_value=50, max_value=300, value=120)

analyze_btn = st.button("🔍 Analyze & Generate Story", type="primary")

sentiment = "neutral"
story_text = None

if analyze_btn and user_input:
    if not pipeline:
        st.error("Sentiment model not loaded. Please fix the file path and restart the app.")
    else:
        with st.spinner("Analyzing sentiment and crafting a story..."):
            try:
                # Use the cached functions
                processed_text = preprocess_text(user_input)
                prediction = pipeline.predict([processed_text])
                sentiment = 'positive' if prediction[0] == 4 else 'negative'
                story_text = generate_story(sentiment, word_limit)
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)
                sentiment = "neutral"
                story_text = None

inject_theme(sentiment)

# Output area
if analyze_btn and user_input and story_text:
    st.subheader("Detected Sentiment")
    sentiment_badge(sentiment)
    st.markdown("---")
    st.subheader("✨ Your Story")
    st.markdown(f"<div class='themed-text themed-accent'>{story_text}</div>", unsafe_allow_html=True)
elif not user_input and analyze_btn:
    st.warning("Please enter some text to analyze.")
else:
    st.info("Your story will appear here after you click **Analyze & Generate Story**.")
