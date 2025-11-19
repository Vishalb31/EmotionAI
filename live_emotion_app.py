import av
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

st.set_page_config(page_title="Live Emotion AI", page_icon="🎭", layout="centered")

# CSS for styling
st.markdown("""
<style>
body {background-color: #0E1117;}
.title {
    font-size: 50px;
    font-weight: 900;
    text-align: center;
    color: white;
}
.subtitle {
    text-align: center;
    color: #AAAAAA;
    font-size: 18px;
}
.emotion-box {
    background-color: rgba(0,0,0,0.4);
    padding: 10px 25px;
    border-radius: 15px;
    display: inline-block;
}
.emoji {
    font-size: 90px;
    animation: bounce 1.2s infinite;
}
@keyframes bounce {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0px); }
}
.motiv-box {
    margin-top:20px;
    text-align:center;
    background: linear-gradient(135deg, #1F2933, #3E4C59);
    padding: 20px;
    border-radius: 15px;
    font-size: 22px;
    color: white;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎭 Live Emotion Detection</div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Detect your facial emotion in real time with AI + Emojis</p>", unsafe_allow_html=True)

emoji_map = {
    "happy": "😀",
    "sad": "😢",
    "angry": "😡",
    "surprise": "😲",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢"
}

motivational_quotes = {
    "happy": "Keep smiling — your happiness spreads positivity ✨",
    "sad": "It's okay to feel sad. Better days are coming 💙",
    "angry": "Take a deep breath — peace begins with you 🌿",
    "surprise": "Life is full of beautiful surprises 🌟",
    "neutral": "Stay calm, balanced, and present 🧘",
    "fear": "Courage is feeling fear but moving forward anyway 💪",
    "disgust": "Release negativity, choose peace 🌱"
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion = "neutral"
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        try:
            if self.frame_count % 10 == 0:
                result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
                self.emotion = result[0]['dominant_emotion']

            cv2.putText(img, f"{self.emotion.upper()} {emoji_map.get(self.emotion,'')}",
                        (30,70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2)
        except:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="Emotion-AI",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

# Motivational message box
if ctx and ctx.state.playing:
    emotion_value = ctx.video_processor.emotion
    st.markdown(
        f"<div class='emotion-box'> <span class='emoji'>{emoji_map.get(emotion_value)}</span></div>",
        unsafe_allow_html=True)

    st.markdown(
        f"<div class='motiv-box'>{motivational_quotes.get(emotion_value)}</div>",
        unsafe_allow_html=True)
