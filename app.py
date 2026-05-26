import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import streamlit as st
import cv2

# Use explicit import path — bypasses mp.solutions lazy attribute chain
# which fails on Streamlit Cloud when the solutions subpackage isn't
# registered as an attribute of the top-level mediapipe module.
import mediapipe as mp
from mediapipe.python.solutions.hands import Hands as _MPHands
from mediapipe.python.solutions.hands import HAND_CONNECTIONS as _MP_HAND_CONNECTIONS

import numpy as np
import joblib
from groq import Groq
from gtts import gTTS
import tempfile
import time
import threading
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Force-load MediaPipe hands module in the main thread NOW so that
# the lazy-loader does not race/fail when called from a WebRTC background thread.
_MP_HANDS = mp.solutions.hands

# --- STYLING & PREMIUM CUSTOM THEME ---
st.set_page_config(
    page_title="ASL Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium CSS for a dark mode interface with glowing cards and micro-animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* General styles */
    .stApp {
        background-color: #0e1117;
        font-family: 'Outfit', sans-serif;
        color: #f0f2f6;
    }
    
    /* Glowing Title & Header */
    .glowing-title {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        text-shadow: 0px 0px 20px rgba(0, 242, 254, 0.2);
    }
    
    /* Elegant card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        margin-bottom: 1.5rem;
    }
    
    /* Glow highlights for specific panels */
    .glow-panel-left {
        border-left: 5px solid #00f2fe;
    }
    .glow-panel-right {
        border-left: 5px solid #00e676;
    }
    
    /* Big predictions style */
    .huge-letter {
        font-size: 5rem;
        font-weight: 800;
        color: #00f2fe;
        text-align: center;
        margin: 0;
        line-height: 1;
        text-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
    }
    
    /* Custom subheaders */
    .panel-subheader {
        font-weight: 600;
        font-size: 1.2rem;
        color: #8a9ba8;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }

    /* Streamlit overrides */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 15px;
    }
</style>
""", unsafe_allow_html=True)


# --- INITIALIZE STATE ---
if "current_word" not in st.session_state:
    st.session_state.current_word = ""
if "current_sentence" not in st.session_state:
    st.session_state.current_sentence = ""
if "groq_output" not in st.session_state:
    st.session_state.groq_output = ""
if "last_action" not in st.session_state:
    st.session_state.last_action = ""


# --- LOAD ASL MODEL & LABELS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("asl_model.pkl")
        labels = joblib.load("labels.pkl")
        return model, labels
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, labels = load_assets()


# --- GET GROQ API KEY ---
groq_api_key = None
try:
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_api_key:
    try:
        from console import groq_api
        groq_api_key = groq_api
    except ImportError:
        pass

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
else:
    groq_client = None


# --- MEDIAPIPE FEATURES PIPELINE ---
ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),       # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12), # Middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# Ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),# Pinky
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

COLORS = {
    "Left":  {"line": (255, 180, 0),   "tip": (0, 80, 255),  "joint": (255, 120, 0), "accent": (255, 180, 0)},
    "Right": {"line": (0, 230, 100),   "tip": (0, 0, 255),   "joint": (0, 200, 80),  "accent": (0, 230, 100)},
}

def _angle_between(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def extract_features(landmarks_proto, is_left=False):
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks_proto.landmark],
        dtype=np.float32,
    )
    # Wrist origin center
    wrist = coords[0].copy()
    coords -= wrist
    scale = np.linalg.norm(coords[9]) + 1e-6
    coords /= scale

    # Mirror x if Right hand to align with Left-hand training data representation
    if not is_left:
        coords[:, 0] *= -1

    # Joint angles
    angles = np.array(
        [_angle_between(coords[a] - coords[b], coords[c] - coords[b])
         for a, b, c in ANGLE_TRIPLETS],
        dtype=np.float32,
    )
    return np.concatenate([coords.flatten(), angles])


# --- HUD & LANDMARK DRAWING FUNCTIONS (MATCHES LETTER_DETECTION.PY) ---
def draw_hand(frame, landmarks_proto, handedness):
    """Draw landmarks with hand-specific colours."""
    h, w = frame.shape[:2]
    c = COLORS.get(handedness, COLORS["Right"])
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_proto.landmark]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], c["line"], 2)
    for i, pt in enumerate(pts):
        r, col = (9, c["tip"]) if i in (4, 8, 12, 16, 20) else (5, c["joint"])
        cv2.circle(frame, pt, r, col, -1)
    # Hand label near wrist
    wx, wy = pts[0]
    tag = "L" if handedness == "Left" else "R"
    cv2.putText(frame, tag, (wx - 12, wy + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, c["accent"], 2)


def draw_hand_panel(frame, x, y, w_panel, handedness, state, is_dominant):
    """Draw a mini panel for one hand's prediction."""
    accent = COLORS.get(handedness, COLORS["Right"])["accent"]
    label_tag = "L" if handedness == "Left" else "R"

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w_panel, y + 68), (15, 15, 15), -1)
    # Dominant indicator — highlight border
    if is_dominant:
        cv2.rectangle(overlay, (x, y), (x + w_panel, y + 68), accent, 2)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Hand label
    cv2.putText(frame, f"{label_tag}", (x + 6, y + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, accent, 2)
    if is_dominant:
        cv2.putText(frame, "★", (x + 24, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if state.stable_letter:
        # Big letter
        cv2.putText(frame, state.stable_letter.upper(), (x + 6, y + 58),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)

        # Confidence bar
        bar_x = x + 55
        bar_w = w_panel - 65
        bar_h = 14
        bar_y = y + 44
        fill  = int(bar_w * state.stable_confidence)
        col   = (0, 220, 0) if state.stable_confidence >= 0.90 else \
                (0, 180, 255) if state.stable_confidence >= 0.75 else (0, 80, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill,  bar_y + bar_h), col, -1)
        cv2.putText(frame, f"{state.stable_confidence:.0%}",
                    (bar_x + 2, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # Raw prediction (small)
        cv2.putText(frame, f"raw: {state.raw_letter.upper()} {state.raw_confidence:.0%}",
                    (bar_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    else:
        cv2.putText(frame, "---", (x + 6, y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)


def draw_hud(frame, left_state, right_state, dominant_hand,
             current_word, current_sentence, fps, hands_detected):
    h, w = frame.shape[:2]

    # ── Top panel background ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (5, 5, 5), -1)
    cv2.rectangle(overlay, (0, h - 100), (w, h), (5, 5, 5), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── Hand panels (top) ──
    panel_w = min(200, w // 2 - 15)
    left_dom  = dominant_hand == "Left"
    right_dom = dominant_hand == "Right"

    draw_hand_panel(frame, 8,       4, panel_w, "Left",  left_state,  left_dom)
    draw_hand_panel(frame, w - panel_w - 8, 4, panel_w, "Right", right_state, right_dom)

    # ── Centre status ──
    if not hands_detected:
        cv2.putText(frame, "Show your hands", (w // 2 - 80, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 255), 1)
    else:
        cv2.putText(frame, "Auto-append Active", (w // 2 - 85, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w // 2 - 22, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 120), 1)

    # ── Bottom panel (word + sentence) ──
    word_disp = current_word if current_word else "_"
    if len(current_sentence) > 55:
        sent_disp = "…" + current_sentence[-55:]
    else:
        sent_disp = current_sentence or "..."

    cv2.putText(frame, f"Word: {word_disp}", (18, h - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 120), 2)
    cv2.putText(frame, f"Sentence: {sent_disp}", (18, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

    # Dominant hand indicator
    dom_tag = f"Dominant: {dominant_hand[0]}"
    cv2.putText(frame, dom_tag, (w - 110, h - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Key hints
    cv2.putText(frame,
                "Hold sign to write | Sync & speak using controls below",
                (12, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)


# --- STABILIZATION HAND STATE CLASS ---
EMA_ALPHA      = 0.45
CONF_THRESHOLD = 0.55
RESET_ENTROPY  = 2.9
VOTE_MAJORITY  = 2
BUFFER_SIZE    = 5

class HandState:
    """Independent EMA + buffer tracking for one hand."""
    def __init__(self):
        self.ema_proba    = np.zeros(len(labels) if labels else 26, dtype=np.float64)
        self.pred_buffer  = deque(maxlen=BUFFER_SIZE)
        self.stable_letter     = ""
        self.stable_confidence = 0.0
        self.raw_letter        = ""
        self.raw_confidence    = 0.0

    def reset(self):
        self.ema_proba[:]      = 0.0
        self.pred_buffer.clear()
        self.stable_letter     = ""
        self.stable_confidence = 0.0
        self.raw_letter        = ""
        self.raw_confidence    = 0.0

    def update(self, proba):
        """Feed in raw model probabilities and update EMA + stabilisation."""
        raw_idx            = int(np.argmax(proba))
        self.raw_letter    = labels[raw_idx]
        self.raw_confidence = float(proba[raw_idx])

        # Entropy-based reset for sign changes
        ent = _entropy(self.ema_proba)
        if ent > RESET_ENTROPY and self.ema_proba.sum() > 0.5:
            self.ema_proba[:] = 0.0
            self.pred_buffer.clear()
            self.stable_letter     = ""
            self.stable_confidence = 0.0

        # EMA blend
        self.ema_proba[:] = EMA_ALPHA * proba + (1 - EMA_ALPHA) * self.ema_proba

        # Confidence gate
        ema_idx  = int(np.argmax(self.ema_proba))
        ema_conf = float(self.ema_proba[ema_idx])

        if ema_conf >= CONF_THRESHOLD:
            self.pred_buffer.append(labels[ema_idx])

        # Voting
        if len(self.pred_buffer) >= VOTE_MAJORITY:
            counts = Counter(self.pred_buffer)
            top_ltr, top_cnt = counts.most_common(1)[0]
            if top_cnt >= VOTE_MAJORITY:
                self.stable_letter     = top_ltr
                self.stable_confidence = ema_conf


# --- WEBRTC ASL PROCESSING CLASS ---
class ASLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.current_word = ""
        self.current_sentence = ""
        self.last_added_letter = ""
        self.dominant_hand = "Right"
        self.no_hand_frames = 0
        self.hands_detected = False
        
        # FPS estimation
        self.fps_timer = time.perf_counter()
        self.fps_count = 0
        self.fps = 0.0
        
        # Hand tracking states
        self.left_state = HandState()
        self.right_state = HandState()
        
        # MediaPipe Hands — using direct import, no mp.solutions attribute chain
        self.hands = _MPHands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame):
        # 1. Convert video frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # 2. Mirror frame horizontally for user friendly view
        img = cv2.flip(img, 1)
        
        # 3. Calculate FPS
        self.fps_count += 1
        now = time.perf_counter()
        if now - self.fps_timer >= 0.5:
            self.fps = self.fps_count / (now - self.fps_timer)
            self.fps_count = 0
            self.fps_timer = now
            
        # 4. MediaPipe Processing
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        
        detected = {"Left": None, "Right": None}
        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, info in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Flipping the frame before MediaPipe means Left hand maps to Left, Right to Right
                hand_label = info.classification[0].label # "Left" or "Right"
                detected[hand_label] = lm
                
        self.hands_detected = any(v is not None for v in detected.values())
        
        # 5. Process left & right hands
        for side, lm_proto in detected.items():
            state = self.left_state if side == "Left" else self.right_state
            if lm_proto is not None:
                # Draw custom landmarks on frame
                draw_hand(img, lm_proto, side)
                
                # Feature engineering + prediction
                is_left = (side == "Left")
                feat = extract_features(lm_proto, is_left).reshape(1, -1)
                
                if model is not None:
                    proba = model.predict_proba(feat)[0]
                    state.update(proba)
            else:
                state.reset()
                
        # 6. Thread-safe word and sentence builder logic
        with self.lock:
            dom_state = self.left_state if self.dominant_hand == "Left" else self.right_state
            if dom_state.stable_letter and dom_state.stable_letter != self.last_added_letter:
                letter = dom_state.stable_letter
                if letter == "space":
                    if self.current_word:
                        self.current_sentence += self.current_word + " "
                        self.current_word = ""
                elif letter == "del":
                    self.current_word = self.current_word[:-1]
                elif letter == "nothing":
                    pass
                else:
                    self.current_word += letter
                self.last_added_letter = letter
                
            # Reset last added letter when dominant hand is removed
            if detected[self.dominant_hand] is None:
                self.last_added_letter = ""
                self.no_hand_frames += 1
            else:
                self.no_hand_frames = 0
                
        # 7. Draw HUD panels directly on the image frame
        draw_hud(img, self.left_state, self.right_state, self.dominant_hand,
                 self.current_word, self.current_sentence, self.fps, self.hands_detected)
                         
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- STREAMLIT UI LAYOUT ---

# Top Banner
st.markdown('<div class="glowing-title">🤟 ASL Fingerspelling & Speech Assistant</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#8a9ba8; font-size:1.15rem; margin-top:-5px; margin-bottom:2rem;">Translating fingerspelled sign language into fluent spoken English in real-time</p>', unsafe_allow_html=True)

# Define columns
col_cam, col_control = st.columns([1.1, 0.9])

with col_cam:
    st.markdown('<div class="glass-card glow-panel-left">', unsafe_allow_html=True)
    st.markdown('<div class="panel-subheader">📷 Real-time Video Stream</div>', unsafe_allow_html=True)
    
    # Render Streamlit-WebRTC camera feed
    ctx = webrtc_streamer(
        key="asl",
        video_processor_factory=ASLVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Instruction card beneath camera
    st.info("💡 **Webcam Instructions**: Start your camera above. Align your hand inside the frame. The HUD overlays show your predictions, confidence, and letters in real-time. Keep a sign stable for ~2 frames to append it!")
    st.markdown('</div>', unsafe_allow_html=True)


# --- PROCESSOR DATA SYNCHRONIZATION ---
if ctx.video_processor:
    # Read active thread states safely with locks
    with ctx.video_processor.lock:
        current_word = ctx.video_processor.current_word
        current_sentence = ctx.video_processor.current_sentence
        hands_detected = ctx.video_processor.hands_detected
else:
    # Fallback to local session states if camera not active
    current_word = st.session_state.current_word
    current_sentence = st.session_state.current_sentence
    hands_detected = False


# --- CONTROLS AND INTERACTION HANDLERS ---
def handle_space():
    # Sync processor
    if ctx.video_processor:
        with ctx.video_processor.lock:
            if ctx.video_processor.current_word:
                ctx.video_processor.current_sentence += ctx.video_processor.current_word + " "
                ctx.video_processor.current_word = ""
    # Sync fallback
    if st.session_state.current_word:
        st.session_state.current_sentence += st.session_state.current_word + " "
        st.session_state.current_word = ""
    st.session_state.last_action = "Appended Space"

def handle_backspace():
    if ctx.video_processor:
        with ctx.video_processor.lock:
            if ctx.video_processor.current_word:
                ctx.video_processor.current_word = ctx.video_processor.current_word[:-1]
            elif ctx.video_processor.current_sentence:
                parts = ctx.video_processor.current_sentence.strip().split(" ")
                ctx.video_processor.current_word = parts[-1]
                ctx.video_processor.current_sentence = " ".join(parts[:-1]) + " " if len(parts) > 1 else ""
    
    if st.session_state.current_word:
        st.session_state.current_word = st.session_state.current_word[:-1]
    elif st.session_state.current_sentence:
        parts = st.session_state.current_sentence.strip().split(" ")
        st.session_state.current_word = parts[-1]
        st.session_state.current_sentence = " ".join(parts[:-1]) + " " if len(parts) > 1 else ""
    st.session_state.last_action = "Deleted last character"

def handle_clear():
    if ctx.video_processor:
        with ctx.video_processor.lock:
            ctx.video_processor.current_word = ""
            ctx.video_processor.current_sentence = ""
            ctx.video_processor.left_state.reset()
            ctx.video_processor.right_state.reset()
    st.session_state.current_word = ""
    st.session_state.current_sentence = ""
    st.session_state.groq_output = ""
    st.session_state.last_action = "Cleared all construction stats"


with col_control:
    st.markdown('<div class="glass-card glow-panel-right">', unsafe_allow_html=True)
    st.markdown('<div class="panel-subheader">⚙️ Configuration & Signing</div>', unsafe_allow_html=True)
    
    # Dominant signing hand radio select (updates background processor real-time)
    selected_dom = st.radio(
        "Dominant Signing Hand:", 
        ["Right", "Left"], 
        horizontal=True, 
        index=0 if (ctx.video_processor and ctx.video_processor.dominant_hand == "Right") else 1 if (ctx.video_processor and ctx.video_processor.dominant_hand == "Left") else 0,
        help="Switches the stabilization focus to the selected hand."
    )
    
    if ctx.video_processor:
        with ctx.video_processor.lock:
            ctx.video_processor.dominant_hand = selected_dom
            
    st.markdown('</div>', unsafe_allow_html=True)

    # Word and sentence building state
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-subheader">📝 Construction Progress</div>', unsafe_allow_html=True)
    
    metric_word = current_word if current_word else "_"
    metric_sent = current_sentence if current_sentence else "..."
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.metric("Current Word", metric_word)
    with c_m2:
        st.metric("Accumulated Sentence", metric_sent)
        
    # Extra word building controls
    c_b1, c_b2, c_b3 = st.columns(3)
    with c_b1:
        st.button("␣ Add Space", use_container_width=True, on_click=handle_space)
    with c_b2:
        st.button("⌫ Backspace", use_container_width=True, on_click=handle_backspace)
    with c_b3:
        st.button("🗑️ Clear", use_container_width=True, on_click=handle_clear)
            
    st.markdown('</div>', unsafe_allow_html=True)

    # Clean & Speak Area
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-subheader">🗣️ Speech & Translation Output</div>', unsafe_allow_html=True)
    
    full_text = (current_sentence + " " + current_word).strip()
    
    if st.button("🔊 Clean & Speak Sentence", use_container_width=True, type="secondary"):
        if not full_text:
            st.warning("Please sign a word or sentence first!")
        else:
            with st.spinner("AI is reconstructing sentence..."):
                if groq_client:
                    try:
                        response = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {
                                    "role": "system",
                                    "content": """You are an legendary, ultra-intelligent AI communication assistant translating raw fingerspelled ASL text for a deaf or speech-impaired user.

The input is highly corrupted text generated live from a sign-recognition computer vision model. It has letter repetitions, missing vowels, visual spelling typos, and no punctuation.

Your absolute core mission:
- Deduplicate repeated letters and correct visual/spelling typos contextually.
- Intelligently reconstruct the complete, highly natural, beautifully polished spoken English message.
- Maintain the user's intent. If they signed shorthand like "GD MRNG", output "Good morning."
- Return ONLY the final clean sentence. NEVER write any introduction, explanation, quotes, or meta-commentary."""
                                },
                                {
                                    "role": "user",
                                    "content": full_text
                                }
                            ],
                            temperature=0.25,
                            max_tokens=128
                        )
                        st.session_state.groq_output = response.choices[0].message.content.strip()
                    except Exception as e:
                        st.error(f"Groq API error: {e}")
                        st.session_state.groq_output = full_text
                else:
                    st.session_state.groq_output = full_text
                
                # Speak via gTTS
                try:
                    tts = gTTS(st.session_state.groq_output, lang="en")
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        tts.save(f.name)
                        st.audio(f.name, format="audio/mp3", autoplay=True)
                except Exception as e:
                    st.error(f"TTS Error: {e}")
                    
    # Display translation result in an elegant success banner
    if st.session_state.groq_output:
        st.success(f"**Polished English**: {st.session_state.groq_output}")
        
    st.markdown('</div>', unsafe_allow_html=True)
