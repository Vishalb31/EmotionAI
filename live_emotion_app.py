import av
import cv2
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Live Emotion Detector", page_icon="🎭")
st.title("🎭 Live Emotion Detection")
st.write("Allow camera access and show your face to detect emotion in real-time!")

emoji_map = {
    "happy": "😀",
    "sad": "😢",
    "angry": "😡",
    "surprise": "😲",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢"
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
                analysis = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
                self.emotion = analysis[0]["dominant_emotion"]

            cv2.putText(
                img,
                f"{self.emotion.upper()} {emoji_map.get(self.emotion, '')}",
                (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )
        except:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="emotion-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
