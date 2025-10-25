import av
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# Streamlit setup
st.set_page_config(page_title="Live Emotion Detector", page_icon="🎭")
st.title("🎭 Live Emotion Detection (AI + Emojis)")
st.write("Allow camera access to detect your facial emotion in real time!")

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

# WebRTC config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Emotion detection processor
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_emotion = "neutral"
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        try:
            # Analyze every 10th frame (for speed)
            if self.frame_count % 10 == 0:
                result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                self.last_emotion = dominant_emotion

            # Draw emotion + emoji on screen
            cv2.putText(img, f"{self.last_emotion.upper()} {emoji_map.get(self.last_emotion, '')}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print("Error:", e)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Live webcam stream
webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,  # ✅ send and receive video
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
