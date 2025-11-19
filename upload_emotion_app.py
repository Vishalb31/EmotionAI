import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Emotion AI (Upload)", page_icon="🎭", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {background-color: #0e1117; color: white;}
    .title {font-size: 48px; font-weight: 800; text-align: center; color: #ffffff;}
    .subtitle {font-size: 20px; text-align: center; opacity: 0.8;}
    .emoji {font-size: 80px; text-align: center; padding: 10px;}
    .result-box {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='title'>🎭 Emotion Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a photo and let AI understand your emotion ❤️</div>", unsafe_allow_html=True)

emojis = {
    "happy": "😀",
    "sad": "😢",
    "angry": "😡",
    "surprise": "😲",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢"
}

motivational_quotes = {
    "happy": "Keep smiling! Your happiness is contagious ✨",
    "sad": "It’s okay to feel sad 🕊 Better days are coming 💖",
    "angry": "Take a deep breath 😌 Let peace guide you 🌿",
    "surprise": "Life is full of wonderful surprises 🌟",
    "neutral": "Stay calm and balanced 🧘‍♂️",
    "fear": "Courage doesn’t mean no fear—keep going 💪",
    "disgust": "Let go of negativity and choose peace 🌱"
}

uploaded_img = st.file_uploader("📸 Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img = Image.open(uploaded_img)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing emotion... ⏳"):
        result = DeepFace.analyze(img_np, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

    st.markdown(f"<div class='emoji'>{emojis.get(emotion, '')}</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='result-box'><h2>{emotion.upper()}</h2><p>{motivational_quotes.get(emotion)}</p></div>",
        unsafe_allow_html=True
    )
