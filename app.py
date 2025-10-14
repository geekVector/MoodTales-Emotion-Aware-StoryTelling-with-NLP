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


#api keys
weather_api_key = "c87ed51b805675cdc2eababdb7cc294b"
google_api_key="AIzaSyAT4Tmn5UTjDt92kbPOPAKf_L5iINbyrNk"

# --- AI and Model Configuration ---

# Securely configure the Google Gemini API
try:
    genai.configure(api_key=st.secrets["google_api_key"])
except Exception as e:
    st.error("Google API Key not found. Please add it to your Streamlit secrets.", icon="🔑")

# --- Dictionaries for Mood Customization ---

color_palette = {
    "Happy":   ["#D8E614", "#E69DB8", "#1AB2E0", "#0CEA40"],
    "Sad":     ["#222831", "#174CA0", "#483518", "#24262C"],
}
mood_emojis = {
    "Happy": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f60a/512.gif",
    "Sad": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f622/512.gif",
    "Neutral": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f610/512.gif"
}
utility_emojis = {
    "clock": "https://fonts.gstatic.com/s/e/notoemoji/latest/23f0/512.gif",
    "globe": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f30d/512.gif",
    "rainyCloud": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f327/512.gif",
    "lightningCloud": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f329/512.gif"
}

# --- Page and Session State Initialization ---

st.set_page_config(layout="wide", page_title="Mood Adaptive Story")

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

# --- Asset Loading and Backend Logic ---

@st.cache_data
def get_local_css(file_name):
    try:
        with open(file_name) as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_name}. Please ensure 'style.css' is present.")
        return ""

# REPLACE your apply_mood_theme function with this final version

def apply_mood_theme(mood):
    colors = color_palette.get(mood, ["#000000"])  # default to black

    duration = len(colors) * 3
    
    # --- Generate CSS for background animation ---
    keyframes = "@keyframes moodCycle {\n"
    step = 100 / len(colors)
    for i, color in enumerate(colors):
        keyframes += f"  {i * step}% {{ background: {color}; }}\n"
    keyframes += f"  100% {{ background: {colors[0]}; }}\n}}"

    # --- NEW: Generate CSS for ALL UI elements based on the mood ---
    dark_themes = {"Sad", "Angry", "Fear", "Disgust"}
    
    # Define common styles for consistent box shapes
    story_box_base_style = """
        border-radius: 10px; padding: 20px; margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15); font-size: 1.1em; line-height: 1.6;
        transition: background-color 0.5s ease, color 0.5s ease, border 0.5s ease;
    """
    info_box_base_style = "border-radius: 10px; transition: background-color 0.5s ease, border 0.5s ease;"

    # Selectors for all text elements we want to change
    text_selectors = "h1, h2, h3, .info-text, #greetingMessage, #moodDisplay, label, .footer-heading, .footer-description, div[data-testid='stMarkdown'] p"

    story_box_style = ""
    ui_style = ""

    if mood in dark_themes:
        # --- Styles for DARK themes ---
        story_box_style = f"""
        .story-container {{
            background-color: rgba(230, 230, 230, 0.9);
            color: #1c1c1c;
            border: 1px solid #777;
            {story_box_base_style}
        }}"""
        ui_style = f"""
        {text_selectors} {{ color: #000000; transition: color 0.5s ease; }}
        .info-box {{ background-color: rgba(0, 0, 0, 0.25); border: 1px solid #888; {info_box_base_style} }}
        """
    else:
        # --- Styles for LIGHT themes ---
        story_box_style = f"""
        .story-container {{
            background-color: rgba(255, 255, 255, 0.7);
            color: #2a2a2a;
            border: 1px solid #ccc;
            {story_box_base_style}
        }}"""
        ui_style = f"""
        {text_selectors} {{ color: #1E1E1E; transition: color 0.5s ease; }}
        .info-box {{ background-color: rgba(255, 255, 255, 0.5); border: 1px solid #ddd; {info_box_base_style} }}
        """
    
    # --- Combine all styles and apply to the app ---
    final_css = f"""
    <style>
        {keyframes}
        {story_box_style}
        {ui_style}
        .stApp {{ animation: moodCycle {duration}s ease-in-out infinite; }}
    </style>
    """
    st.markdown(final_css, unsafe_allow_html=True)
@st.cache_data(show_spinner="Fetching live data...")
def get_live_data():
    try:
        loc_res = requests.get("http://ip-api.com/json/", timeout=5)
        loc_res.raise_for_status()
        loc_data = loc_res.json()
        city, country = loc_data.get("city", "Unknown"), loc_data.get("country", "")
        weather_api_key = weather_api_key
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
    prompt = prompt = (
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
        model = genai.GenerativeModel('gemini-2.0-flash')  # ✅ updated model name
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
col1, col2, col3 = st.columns(3, gap="large")
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
    
    mood_col1, mood_col2 = st.columns([1, 4])
    with mood_col1:
        st.markdown(f"<div class='emoji-box'><img src='{st.session_state.mood_emoji_url}' width='100'></div>", unsafe_allow_html=True)
    with mood_col2:
        st.markdown(f"<div id='moodDisplay'>Your current mood is: <strong>{st.session_state.mood}</strong></div>", unsafe_allow_html=True)

    mood_text = st.text_area("Type something about your mood...", placeholder=f"How are you feeling today, {st.session_state.username}?", height=100)
    word_limit = st.slider(
    "Select your desired story length (in words):",
    min_value=50,
    max_value=400,
    value=150,
    step=10,
    help="Adjust the slider to control how long your story will be."
)

    if st.button("✨ Generate Story ✨", use_container_width=True):
        if mood_text and pipeline:
            with st.spinner("Analyzing your mood and crafting a story..."):
                processed_text = preprocess_text(mood_text)
                prediction = pipeline.predict([processed_text])
                sentiment = 'Happy' if prediction[0] == 4 else 'Sad'
                
                st.session_state.mood = sentiment
            st.session_state.story = story_generation(sentiment, word_limit)
            st.session_state.mood_emoji_url = mood_emojis.get(st.session_state.mood, mood_emojis["Neutral"])
            st.rerun()  # ✅ instant theme + story refresh
        elif not mood_text:
            st.warning("Please type something about your mood first.", icon="✍️")
        else:
            st.error("Sentiment model could not be loaded. Please check the logs.")

    st.markdown("---")
    st.subheader("Your Adaptive Story")
    st.markdown(f"<div class='story-container'>{st.session_state.story}</div>", unsafe_allow_html=True)
st.markdown("""
<style>
/* Change text color inside Streamlit text input */
[data-testid="stTextInput"] input {
    color: white;           /* Text color */
    background-color: #222; /* Optional: background color */
    font-weight: bold;       /* Optional styling */
}
</style>
""", unsafe_allow_html=True)

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
  mix-blend-mode: difference; /* auto inverse based on bg */
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
  mix-blend-mode: difference; /* makes text inverse automatically */
}

/* All general text (paragraphs, labels, etc.) */
.dynamic-text {
  mix-blend-mode: difference;
  transition: color 0.6s ease;
}

/* Optional hover for better visibility */
.story-output:hover, .dynamic-heading:hover {
  text-shadow: 0 0 10px rgba(255,255,255,0.4);
}

</style>
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
""", unsafe_allow_html=True)
# Pick a representative color for the current mood
bg_color = color_palette.get(st.session_state.mood, ["#ffffff"])[0]

# Inject JavaScript with the selected color
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
</style>

<footer class="dynamic-footer">
    <div class="footer-content">
        <h2 class="footer-heading">🌟 Mood Adaptive Story Generator</h2>
        <p class="footer-description" style="color: black;">Dive into a world where every story resonates with your emotions. This app’s theme, tone, and narrative content adapt seamlessly to your current feelings — whether you’re joyful, melancholic, angry, or curious. Powered by advanced AI, it crafts personalized tales that reflect your mood, turning your emotions into immersive storytelling experiences. 💫</p>
    </div>
</footer>
""", unsafe_allow_html=True)

