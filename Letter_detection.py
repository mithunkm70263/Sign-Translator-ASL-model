import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import subprocess
import groq
import platform
import time
from collections import deque, Counter

# Load Groq API Key and initialize client
try:
    from console import groq_api
except ImportError:
    groq_api = None

if groq_api:
    groq_client = groq.Groq(api_key=groq_api)
else:
    groq_client = None


try:
    model  = joblib.load("asl_model.pkl")
    labels = joblib.load("labels.pkl")
    print(f"Model loaded — {len(labels)} classes: {labels}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

N_CLASSES = len(labels)


ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),       # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12), # Middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# Ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),# Pinky
]


def _angle_between(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def extract_features(hand_landmarks_proto, mirror_x=False):
    """
    Extract 78-dim feature vector from hand landmarks.
    mirror_x=True flips the x-axis so inference can handle mirrored cameras.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks_proto.landmark],
        dtype=np.float32,
    )

    # Wrist-relative normalisation
    wrist = coords[0].copy()
    coords -= wrist
    scale = np.linalg.norm(coords[9]) + 1e-6
    coords /= scale

    if mirror_x:
        coords[:, 0] *= -1

    # Joint angle features (15)
    angles = np.array(
        [_angle_between(coords[a] - coords[b], coords[c] - coords[b])
         for a, b, c in ANGLE_TRIPLETS],
        dtype=np.float32,
    )

    return np.concatenate([coords.flatten(), angles])  # (78,)


def predict_best_orientation(hand_landmarks_proto):
    """Try both webcam orientations and keep the model's stronger prediction."""
    normal = extract_features(hand_landmarks_proto, mirror_x=False).reshape(1, -1)
    mirrored = extract_features(hand_landmarks_proto, mirror_x=True).reshape(1, -1)

    normal_proba = model.predict_proba(normal)[0]
    mirrored_proba = model.predict_proba(mirrored)[0]

    if float(np.max(mirrored_proba)) > float(np.max(normal_proba)):
        return mirrored_proba
    return normal_proba


def clean_with_groq(raw_text):
    """
    Cleans raw, heavily corrupted ASL fingerspelling text into natural spoken English
    using a state-of-the-art context-aware prompt on Llama 3.1.
    """
    if not groq_client:
        print("[Groq API client not initialized. Using raw text.]")
        return raw_text.strip()
        
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are an legendary, ultra-intelligent AI communication assistant translating raw fingerspelled ASL text for a deaf or speech-impaired user.

The input is highly corrupted text generated live from a sign-recognition computer vision model. It has:
1. Meaningless letter repetitions (e.g., "HHHIIII HOWWW ARRRE YYOOOUUU")
2. Missing vowels & phonetic spelling shortcuts (e.g., "HW R U", "THK U", "PLZ", "WTR", "BTHRM")
3. Typical visual typos caused by model limitations swapping hand signs with similar configurations (e.g., A/E/S/O/T or M/N/T).
4. No capitalization, missing grammar/articles, or punctuation.

Your absolute core mission:
- Deduplicate repeated letters and correct visual/spelling typos contextually.
- Intelligently reconstruct the complete, highly natural, beautifully polished spoken English message.
- Maintain the user's intent. If they signed shorthand like "GD MRNG", output "Good morning." If they signed "HLPP MEE PLS", output "Help me, please."
- Ensure it sounds natural and flowy, formatted perfectly for an audio text-to-speech speaker.
- NEVER explain, never add preamble, never use quotes or notes. ONLY output the single corrected final sentence.

Examples:
Input: "HII HOWW R UU" ➔ Output: "Hi, how are you?"
Input: "I M HNGRY" ➔ Output: "I am hungry."
Input: "GD MRNG" ➔ Output: "Good morning."
Input: "HLPP MEE PLS" ➔ Output: "Help me, please."
Input: "I ND WTTR" ➔ Output: "I need water."
Input: "THK UU VRII MCCH" ➔ Output: "Thank you very much."
Input: "WHRR IS THT BTHRM" ➔ Output: "Where is the bathroom?"
Input: "NICE TO MET YU" ➔ Output: "Nice to meet you."
Input: "PLSS CL TCHRR" ➔ Output: "Please call the teacher." """
                },
                {
                    "role": "user",
                    "content": raw_text
                }
            ],
            temperature=0.25,
            max_tokens=128
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Groq API Error] {e}")
        return raw_text.strip()


def speak_cleaned_sentence(raw_text):
    """
    Spawns a background thread to call Groq to clean the sentence and play the TTS.
    This guarantees 0% lag or stuttering in the live camera window!
    """
    def worker():
        cleaned = clean_with_groq(raw_text)
        print(f"\n[Groq Correction] Raw: '{raw_text}' -> Cleaned: '{cleaned}'")
        tts.speak(cleaned)

    threading.Thread(target=worker, daemon=True).start()


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,               # ← TWO HANDS
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)


class CameraThread:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock  = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            return (self.frame is not None), (
                self.frame.copy() if self.frame is not None else None
            )

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()



class TTSManager:
    """Speaks text in a background process so it never blocks the main loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._is_mac = platform.system() == "Darwin"

    def speak(self, text):
        text = text.strip()
        if not text:
            return
        with self._lock:
            # Kill any ongoing speech first
            if self._process and self._process.poll() is None:
                self._process.terminate()
            if self._is_mac:
                self._process = subprocess.Popen(
                    ["say", "-r", "180", text],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                # Fallback: pyttsx3 in a thread
                threading.Thread(target=self._pyttsx_speak, args=(text,), daemon=True).start()

    def _pyttsx_speak(self, text):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 180)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS fallback error] {e}")

    def is_speaking(self):
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def stop(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()


tts = TTSManager()



EMA_ALPHA      = 0.45
CONF_THRESHOLD = 0.55
RESET_ENTROPY  = 2.9
VOTE_MAJORITY  = 2
BUFFER_SIZE    = 5


class HandState:
    """Independent EMA + buffer tracking for one hand."""

    def __init__(self):
        self.ema_proba    = np.zeros(N_CLASSES, dtype=np.float64)
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


def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


# Create states for left and right hands
left_state  = HandState()
right_state = HandState()



current_word        = ""
current_sentence    = ""
last_added_letter   = ""
no_hand_frames      = 0
dominant_hand       = "Right"        # user can toggle with 'd'
auto_speak_frames   = 90             # ~3 sec at 30 fps
spoken_sentence     = ""             # track what was already spoken



# Hand-specific colours
COLORS = {
    "Left":  {"line": (255, 180, 0),   "tip": (0, 80, 255),  "joint": (255, 120, 0), "accent": (255, 180, 0)},
    "Right": {"line": (0, 230, 100),   "tip": (0, 0, 255),   "joint": (0, 200, 80),  "accent": (0, 230, 100)},
}


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
             current_word, current_sentence, fps, speaking, hands_detected):
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
    elif speaking:
        # Pulsing speaker indicator
        pulse = int(128 + 127 * np.sin(time.time() * 6))
        cv2.putText(frame, "Speaking...", (w // 2 - 55, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, pulse, 255), 2)

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
                "[Space]=word  [Bksp]=del  [Enter]=speak  [d]=swap hand  [c]=clear  [q]=quit",
                (12, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 100, 100), 1)



cam = CameraThread(0)

print("═" * 56)
print("  ASL Sign Recognition — Two-Hand Legendary Edition")
print("═" * 56)
print("  [Space]  Add word to sentence")
print("  [Enter]  Speak sentence aloud")
print("  [Bksp]   Delete last letter")
print("  [d]      Toggle dominant hand (L↔R)")
print("  [c]      Clear everything")
print("  [q]      Quit")
print("═" * 56)

fps_timer  = time.perf_counter()
fps_count  = 0
fps        = 0.0

while True:
    ok, frame = cam.read()
    if not ok or frame is None:
        time.sleep(0.005)
        continue

    # FPS
    fps_count += 1
    now = time.perf_counter()
    if now - fps_timer >= 0.5:
        fps = fps_count / (now - fps_timer)
        fps_count = 0
        fps_timer = now

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    # Identify detected hands 
    detected = {"Left": None, "Right": None}   # landmark proto per hand

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # MediaPipe reports handedness from camera perspective,
            # and the displayed image is mirrored, so swap back to the user's hand.
            hand_label = info.classification[0].label   # "Left" or "Right"
            hand_label = "Left" if hand_label == "Right" else "Right"
            detected[hand_label] = lm

    hands_detected = any(v is not None for v in detected.values())

    # ── Process each hand ────────────────────────────────────────────────────
    for side, lm_proto in detected.items():
        state = left_state if side == "Left" else right_state

        if lm_proto is not None:
            draw_hand(frame, lm_proto, side)

            proba   = predict_best_orientation(lm_proto)
            state.update(proba)
        else:
            state.reset()

    # ── Word builder — uses dominant hand's stable letter ────────────────────
    dom_state = left_state if dominant_hand == "Left" else right_state

    if dom_state.stable_letter and dom_state.stable_letter != last_added_letter:
        letter = dom_state.stable_letter
        if letter == "space":
            if current_word:
                current_sentence += current_word + " "
                current_word = ""
        elif letter == "del":
            current_word = current_word[:-1]
        elif letter == "nothing":
            pass   # ignore 'nothing' class
        else:
            current_word += letter
        last_added_letter = letter

    # Reset when dominant hand is removed
    if detected[dominant_hand] is None:
        last_added_letter = ""
        no_hand_frames += 1
    else:
        no_hand_frames = 0

    # ── Auto-speak when no hands for ~3 seconds ─────────────────────────────
    full_text = (current_sentence + " " + current_word).strip()
    if (no_hand_frames >= auto_speak_frames
            and full_text
            and full_text != spoken_sentence
            and not tts.is_speaking()):
        speak_cleaned_sentence(full_text)
        spoken_sentence = full_text

    # ── HUD ──────────────────────────────────────────────────────────────────
    draw_hud(frame, left_state, right_state, dominant_hand,
             current_word, current_sentence, fps, tts.is_speaking(), hands_detected)

    cv2.imshow("ASL Sign Recognition", frame)

    # ── Keyboard ─────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):                       # Space → push word
        if current_word:
            current_sentence += current_word + " "
            current_word = ""
            last_added_letter = ""
    elif key == 8 or key == 127:                # Backspace → delete letter
        if current_word:
            current_word = current_word[:-1]
            last_added_letter = ""
    elif key == 13 or key == 10:                # Enter → speak now
        full = (current_sentence + " " + current_word).strip()
        if full:
            speak_cleaned_sentence(full)
            spoken_sentence = full
    elif key == ord('d'):                       # d → toggle dominant hand
        dominant_hand = "Left" if dominant_hand == "Right" else "Right"
        last_added_letter = ""
        print(f"Dominant hand → {dominant_hand}")
    elif key == ord('c'):                       # c → clear everything
        current_word      = ""
        current_sentence  = ""
        last_added_letter = ""
        spoken_sentence   = ""
        left_state.reset()
        right_state.reset()
        tts.stop()

cam.release()
hands.close()
tts.stop()
cv2.destroyAllWindows()
print("Done. Bye!")
