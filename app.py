import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

# Emoji mapping
emoji_map = {
    "happy": "😀",
    "sad": "😢",
    "angry": "😡",
    "surprise": "😲",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢"
}

st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")
st.title("😊 AI Emotion Detector (GenAI + Emoji)")
st.write("Upload your photo to see what emotion you're expressing!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing emotion..."):
        result = DeepFace.analyze(img_path=img_cv, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

    st.success(f"Detected Emotion: **{emotion.upper()}** {emoji_map.get(emotion, '')}")

    # Optional GenAI style motivational text
    if emotion == "happy":
        st.info("💬 Keep smiling! Your positivity is contagious 😄")
    elif emotion == "sad":
        st.info("💬 Don’t worry, even the darkest nights end with sunrise 🌅")
    elif emotion == "angry":
        st.info("💬 Take a deep breath — peace always wins 🕊️")
    elif emotion == "surprise":
        st.info("💬 Wow! You look amazed — keep that curiosity alive 🤩")
    else:
        st.info("💬 Stay calm and confident, you’re doing great 💪")
