import streamlit as st
import time
import requests
from datetime import datetime
import random
import os
import re
import joblib
import nltk
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment
import io


# --- AI and Model Configuration ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Securely configure the Google Gemini API
try:
    genai.configure(api_key="AIzaSyAT4Tmn5UTjDt92kbPOPAKf_L5iINbyrNk")
except Exception as e:
    st.error("Google API Key not found. Please add it to your Streamlit secrets.", icon="🔑")

# --- Dictionaries for Mood Customization ---

color_palette = {
    "Happy":   ["#FFD700", "#FF5722", "#D8E089", "#A84E6C"],
    "Sad":     ["#39546D", "#13143B", "#192D31", "#000000"],
    "Neutral": ["#745F47", "#77B177", "#9566A3", "#4B4B4B"]
}
mood_emojis = {
    "Happy": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHEyZ3BsYnh5MWxnenp1a3l0amplcnlsdHNtZWl0emhndzB6ejhzaSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/11sBLVxNs7v6WA/giphy.gif",
    "Sad": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXIyejVyNXVucHdhczR1OWVxYnM4NTJkZTV5OGttcDlrcDhleXQydyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/fhLgA6nJec3Cw/giphy.gif",
    "Neutral": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExb2VxeWpnNXNydTB2cTJ0cm5nYzhxczE4d3B6bTkzNTdkaTNicjdhOCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/iyCUpd3MOYLf8COyPQ/giphy.gif"
}
utility_emojis = {
    "clock": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExanVlZ2puNjdzd3I5NjUwOW13aXlkd3ljZWxzYzAxa2dsczh4a2Z2dyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/2zdVnsL3mbrs4xg4fr/giphy.gif",
    "globe": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExajhzaHdvMm05a29zcTNibzdjeTdpdHAxYzc5eHc3YmJienloYXk4ZSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/mf8UbIDew7e8g/giphy.gif",
    "rainyCloud": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2t5eWs2aTl4OTJpeTF4MjJ6NDNxNmpxbGFjYjR1eng1Y2p1aW95dyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/gk3s6G7AdUNkey0YpE/giphy.gif",
    "lightningCloud": "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2t5eWs2aTl4OTJpeTF4MjJ6NDNxNmpxbGFjYjR1eng1Y2p1aW95dyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/xaZCqV4weJwHu/giphy.gif"
}

# --- Page and Session State Initialization ---

st.set_page_config(layout="wide", page_title=" AI Mood Adaptive Story")

if 'username' not in st.session_state:
    st.session_state.username = ""
if 'mood' not in st.session_state:
    st.session_state.mood = "Neutral"
    st.session_state.mood_emoji_url = mood_emojis["Neutral"]
    st.session_state.story = "Your story will appear here once you share your mood."
if 'location_data' not in st.session_state:
    st.session_state.location_data = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'heading_index' not in st.session_state:
    st.session_state.heading_index = 0
if 'mood_history' not in st.session_state:
    st.session_state.mood_history = []


# --- Asset Loading and Backend Logic ---

@st.cache_data
def get_local_css(file_name):
    try:
        with open(file_name) as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_name}. Please ensure 'style.css' is present.")
        return ""

# --- UPDATED: apply_mood_theme function for single, high-contrast background emoji ---

def apply_mood_theme(mood):
    emoji_url = mood_emojis.get(mood, mood_emojis["Neutral"])
    
    # Define text colors for maximum contrast
    if mood == "Sad":
        text_color = "#FFFFFF"  # White text for high contrast on dark backgrounds
        story_box_bg = "rgba(0, 0, 0, 0.6)" # Darker story box for legibility
        base_bg_color = "#202020" # Dark base background color
        background_opacity = 0.5 # Increased opacity for higher contrast on dark theme
    else:
        text_color = "#1E1E1E"  # Near-black text for high contrast on light/bright backgrounds
        story_box_bg = "rgba(255, 255, 255, 0.85)" # Lighter, almost opaque story box for legibility
        base_bg_color = "#F0F0F0" # Light base background color
        background_opacity = 0.3 # Increased opacity for higher contrast on light theme
    
    # Define common styles for consistent box shapes
    story_box_base_style = f"""
        border-radius: 10px; padding: 20px; margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5); /* Increased shadow for lift and contrast */
        font-size: 1.1em; line-height: 1.6;
        transition: background-color 0.5s ease, color 0.5s ease, border 0.5s ease;
    """
    info_box_base_style = "border-radius: 10px; transition: background-color 0.5s ease, border 0.5s ease;"

    # Selectors for all text elements we want to change
    text_selectors = "h1, h2, h3, .info-text, #greetingMessage, #moodDisplay, label, .footer-heading, .footer-description, div[data-testid='stMarkdown'] p, .stMarkdown"
    
    # --- Styles for story box and general UI ---
    story_box_style = f"""
        .story-container {{
            background-color: {story_box_bg};
            color: {text_color};
            border: 1px solid {text_color}44; /* Subtle border */
            {story_box_base_style}
        }}"""
    
    ui_style = f"""
        {text_selectors} {{ color: #000000; transition: color 0.5s ease; }}
        .info-box {{ 
            background-color: {story_box_bg.replace('0.85', '0.6').replace('0.6', '0.4')}; 
            border: 1px solid {text_color}33; 
            {info_box_base_style} 
        }}
        /* Ensure the Streamlit markdown container also gets the text color */
        div[data-testid='stText'] {{ color: {text_color}; }}
    """
    
    # --- NEW: Background Image CSS using ::before for contrast control ---
    background_image_css = f"""
        /* Set base background color for the app and ensure position is relative for the pseudo-element */
        [data-testid="stAppViewContainer"] {{
            background-color: {base_bg_color}; 
            position: relative;
            z-index: 1; /* Ensures content is above the pseudo-element */
        }}

        /* Create the pseudo-element to hold the image with low opacity */
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed; /* Fixed so it covers the whole viewport */
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('{emoji_url}');
            background-size: cover; /* CHANGED: To display the image once, covering the screen */
            background-repeat: no-repeat; /* CHANGED: To prevent tiling */
            background-position: center; /* Center the image */
            opacity: {background_opacity}; /* ADJUSTED: Controls the contrast/fade of the background image only */
            z-index: -1; /* Puts the background image behind all content */
            pointer-events: none; /* Allows clicks to go through to the main content */
        }}
    """
    
    # --- Combine all styles and apply to the app ---
    final_css = f"""
    <style>
        {story_box_style}
        {ui_style}
        {background_image_css}
    </style>
    """
    st.markdown(final_css, unsafe_allow_html=True)
apply_mood_theme(st.session_state.mood)

# --- ADD THIS SIDEBAR CODE ---
with st.sidebar:
    st.title("Profile")
    if st.session_state.username:
        st.header(f"🧑‍💻 {st.session_state.username}")
    else:
        st.header("🧑‍💻 Guest")

    st.markdown("---")
    st.subheader("📝 Mood History")
    
    if not st.session_state.mood_history:
        st.info("Your mood entries will appear here.")
    else:
        # Display in reverse order (most recent on top)
        for entry in reversed(st.session_state.mood_history):
                st.markdown(f"> <div style='color: #FFFFFF;'>{entry}</div>", unsafe_allow_html=True)
# --- END OF SIDEBAR CODE ---

@st.cache_data(show_spinner="Fetching live data...")
def get_live_data():
    try:
        loc_res = requests.get("http://ip-api.com/json/", timeout=5)
        loc_res.raise_for_status()
        loc_data = loc_res.json()
        city, country = loc_data.get("city", "Unknown"), loc_data.get("country", "")
        weather_api_key = "c87ed51b805675cdc2eababdb7cc294b"
        if not weather_api_key:
            st.error("Weather API key not found. Please add it to your Streamlit secrets.")
            return f'{city}, {country}', None
        weather_res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric", timeout=5)
        weather_res.raise_for_status()
        return f'{city}, {country}', weather_res.json()
    except Exception as e:
        st.error(f"Could not fetch live data. Error: {e}")
        return "Location unavailable", None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model1.pkl")

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Sentiment model not found at: {MODEL_PATH}")
        st.error("Please make sure 'sentiment_model1.pkl' is in the same folder as app.py.")
        return None

def download_nltk_data():
    packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            nltk.download(package)

def preprocess_text(text):
    sentiment_words = {"not", "no", "nor", "never", "don't", "didn't", "isn't", "wasn't", "won't", "can't", "couldn't"}
    default_stopwords = set(stopwords.words('english'))
    custom_stopwords = default_stopwords - sentiment_words
    text = re.sub(r'https?:\/\/\S+|www\.\S+|@\w+|<@!?[\d]+>', '', text).lower()
    tokens = [word for word in nltk.word_tokenize(text) if word.isalpha() and word not in custom_stopwords]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def story_generation(sentiment, word_limit=150):
    prompt = (
        f"Write a unique, suspenseful short story that instantly captures the reader’s attention in the first paragraph "
        f"with a mysterious or shocking event. The story should revolve around a main character who discovers a hidden truth "
        f"that turns their reality upside down. Introduce escalating layers of suspense, including red herrings, "
        f"unexpected betrayals, and moral dilemmas. Structure the plot with rising tension in every scene, "
        f"ending each major section with a cliffhanger or unanswered question that compels the reader to continue. "
        f"Set the story in an unusual or eerie setting (e.g., abandoned town, remote island, underground facility). "
        f"Create emotionally engaging stakes by tying the mystery to the character’s past or relationships. "
        f"The final twist should reframe everything the reader thought they knew, leaving a lingering question or emotional impact. "
        f"Do not resolve everything neatly—leave room for interpretation. "
        f"Keep the story under {word_limit} words, and ensure the tone is influenced by the user's *{sentiment}* feeling."
    )
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')  # ✅ updated model name
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred during story generation: {e}")
        return f"A story for a '{sentiment}' day is unfolding... (Error connecting to AI Story Generator)."

# --- Main App Execution ---

download_nltk_data()
pipeline = load_sentiment_model()
if st.session_state.location_data is None:
    st.session_state.location_data, st.session_state.weather_data = get_live_data()

apply_mood_theme(st.session_state.mood)
st.markdown(f"<style>{get_local_css('style.css')}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
/* Keeping this block for safety and to override default Streamlit style if necessary */
.story-container {
    background-color: rgba(255, 255, 255, 0.5);
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    font-size: 1.1em;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# Top Info Bar
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="info-box"><img src="{utility_emojis["clock"]}" class="top-bar-emoji"><div class="info-text">{datetime.now().strftime("%d %b, %I:%M %p")}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="info-box"><img src="{utility_emojis["globe"]}" class="top-bar-emoji"><div class="info-text">{st.session_state.location_data}</div></div>', unsafe_allow_html=True)
with col3:
    weather_text, weather_icon = "Weather unavailable", "🌥️"
    if st.session_state.weather_data and "main" in st.session_state.weather_data:
        weather = st.session_state.weather_data
        temp = round(weather["main"]["temp"])
        desc = weather["weather"][0]["description"]
        weather_text = f"{desc.title()}, {temp}°C"
        main_cond = weather["weather"][0]["main"].lower()
        if main_cond in ["rain", "drizzle", "squall"]:
            weather_icon = f'<img src="{utility_emojis["rainyCloud"]}" class="top-bar-emoji">'
        elif main_cond == "thunderstorm":
            weather_icon = f'<img src="{utility_emojis["lightningCloud"]}" class="top-bar-emoji">'
    st.markdown(f'<div class="info-box">{weather_icon}<div class="info-text">{weather_text}</div></div>', unsafe_allow_html=True)

# Main UI
st.markdown(f"<h1  id='dynamicHeading'>AI Mood Adaptive Story</h1>", unsafe_allow_html=True)

if not st.session_state.username:
    st.session_state.username = st.text_input("First, what's your name?", key="name_input")
    if st.session_state.username:
        st.rerun()
else:
    st.markdown(f"<div id='greetingMessage'>Hello {st.session_state.username}, how are you feeling today?</div>", unsafe_allow_html=True)
    
     # 1. Initialize input mode if not exists
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "Type"

    # 2. CREATE THE OPTION SCREEN (The Buttons)
    col_type, col_voice = st.columns(2)
    with col_type:
        if st.button("⌨️ Type Input", use_container_width=True):
            st.session_state.input_mode = "Type"
    with col_voice:
        if st.button("🎙️ Voice Input", use_container_width=True):
            st.session_state.input_mode = "Voice"

    st.markdown("---")

    # 3. SWITCH CASE LOGIC for Input Method
    mood_text = "" # Initialize empty string to store input
    voice_text = "" # Initialize empty string to store voice input
    st.session_state.mood_text = voice_text

    if st.session_state.input_mode == "Type":
        mood_text = st.text_area("Type something about your mood...", 
                                placeholder=f"How are you feeling today, {st.session_state.username}?", 
                                height=100)
    
    elif st.session_state.input_mode == "Voice":

        st.write("Click the microphone to start talking:")

        # 🎤 ALWAYS define audio first
        audio = mic_recorder(
            start_prompt="⏺️ Start Recording",
            stop_prompt="⏹️ Stop",
            key="recorder"
        )

        # 🎧 Process the recorded audio
        if audio is not None and audio.get("bytes"):
            with st.spinner("Processing your voice..."):
                try:
                    audio_segment = AudioSegment.from_file(
                        io.BytesIO(audio["bytes"])
                    )

                    # Convert to WAV
                    wav_io = io.BytesIO()
                    audio_segment.export(wav_io, format="wav")
                    wav_io.seek(0)

                    r = sr.Recognizer()
                    with sr.AudioFile(wav_io) as source:
                        audio_data = r.record(source)

                    voice_text = r.recognize_google(audio_data)

                    # ✅ Save to session state
                    st.session_state.mood_text = voice_text

                    st.success("Speech captured successfully!")
                    st.write("🗣️ **You said:**", voice_text)

                except Exception as e:
                    st.error(f"Voice processing failed: {e}")


    word_limit = st.slider("Select Story Length (in words)", min_value=50, max_value=400, value=150, step=10)

    analyze_text = mood_text if st.session_state.input_mode == "Type" else voice_text
    mood_text= st.session_state.get("mood_text", "").strip()
    voice_text= st.session_state.get("voice_text", "").strip()
    


if st.button("✨ Generate Story ✨", use_container_width=True):

    if analyze_text and pipeline:
        with st.spinner("Analyzing your mood and crafting a story..."):

            processed_text = preprocess_text(analyze_text)
            prediction = pipeline.predict([processed_text])

            sentiment = "Happy" if prediction[0] == 4 else "Sad"

            apply_mood_theme(sentiment)
            st.session_state.mood = sentiment
            st.session_state.story = story_generation(sentiment,word_limit=word_limit)
            st.session_state.mood_emoji_url = mood_emojis.get(sentiment, mood_emojis["Neutral"])
            st.session_state.mood_history.append(analyze_text)
            st.rerun()

    elif not analyze_text:
        st.warning("Please type or speak about your mood first 🎤✍️")

    else:
        st.error("Sentiment model could not be loaded. Please check the logs.")

st.markdown("---")
st.subheader("Your Adaptive Story")
st.markdown(f"<div class='story-container'>{st.session_state.story}</div>", unsafe_allow_html=True)
st.markdown("""
<style>
:root {
  --bg-color: #ffffff;  /* default background */
  --text-color: #000000; /* default text */
}

/* Main container applies background dynamically */
body, .main-container {
  background-color: var(--bg-color);
  color: var(--text-color);
  transition: background-color 0.6s ease, color 0.6s ease;
}

/* Headings */
.dynamic-heading {
  font-weight: bold;
  text-align: center;
  transition: color 0.6s ease;
}

/* Story box container */
.story-output {
  padding: 20px;
  margin-top: 25px;
  border-radius: 15px;
  backdrop-filter: blur(6px);
  font-size: 18px;
  line-height: 1.6;
  font-family: "Georgia", serif;
  transition: background-color 0.6s ease, color 0.6s ease;
  background-color: rgba(255, 255, 255, 0.15);
}

/* All general text (paragraphs, labels, etc.) */
.dynamic-text {
  transition: color 0.6s ease;
}

/* Optional hover for better visibility */
.story-output:hover, .dynamic-heading:hover {
  text-shadow: 0 0 10px rgba(255,255,255,0.4);
}

</style>
<script>
<script>
function setDynamicTheme(bgColor) {
  // Apply background color
  document.documentElement.style.setProperty('--bg-color', bgColor);

  // Compute inverted text color
  const inverted = invertColor(bgColor);
  document.documentElement.style.setProperty('--text-color', inverted);
}

// Function to invert any hex color dynamically
function invertColor(hex) {
  hex = hex.replace('#', '');
  if (hex.length === 3) {
    hex = hex.split('').map(h => h + h).join('');
  }
  const r = (255 - parseInt(hex.substring(0,2), 16)).toString(16).padStart(2, '0');
  const g = (255 - parseInt(hex.substring(2,4), 16)).toString(16).padStart(2, '0');
  const b = (255 - parseInt(hex.substring(4,6), 16)).toString(16).padStart(2, '0');
  return `#${r}${g}${b}`;
}

// Example: mood change or background update
// setDynamicTheme('#222831');  // dark mood
// setDynamicTheme('#FFD0C7');  // happy mood
</script>

</script>
""", unsafe_allow_html=True)
# Get the first color of the current mood
bg_color = color_palette.get(st.session_state.mood, color_palette["Neutral"])[0]

# Then call your JS function
st.markdown(f"<script>setDynamicTheme('{bg_color}');</script>", unsafe_allow_html=True)

# --- NEW: Footer with Box Styling and Hover Animation ---
st.markdown("---")
st.markdown("""
<style>
.dynamic-footer {
    background-color: rgba(255, 255, 255, 0.6); /* Semi-transparent white */
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-top: 40px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease-in-out; /* This makes the hover effect smooth */
}

.dynamic-footer:hover {
    transform: translateY(-5px); /* Lifts the box up slightly */
    box-shadow: 0 8px 16px rgba(0,0,0,0.2); /* Makes the shadow more pronounced */
}

.footer-heading {
    margin-bottom: 10px;
    color: #333;
}

.footer-description {
    color: #555;
    font-size: 1.1em;
}

/* ENSURE FOOTER TEXT CONTRAST */
.dynamic-footer .footer-heading, 
.dynamic-footer .footer-description {
    color: #1E1E1E !important; /* Forces dark color for contrast on light footer background */
}
</style>

<footer class="dynamic-footer">
    <div class="footer-content">
        <h2 class="footer-heading">🌟 Mood Adaptive Story Generator</h2>
        <p class="footer-description">
Experience stories that truly resonate with you — this app’s theme and narrative content adapt to your feelings, moods, and emotions in real time. Powered by AI, bringing your inner world to life. 💫
</p>
    </div>
</footer>
""", unsafe_allow_html=True)







