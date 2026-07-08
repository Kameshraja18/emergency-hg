import cv2
import mediapipe as mp
from gtts import gTTS
import os
import time
import collections
import json
import tempfile
from pathlib import Path
from twilio.rest import Client

try:
    import streamlit as st
except Exception:
    st = None

try:
    import pygame
except Exception:
    pygame = None

# Optional: load variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

"""
Hand-gesture driven assistive tool:
 - Detects number of raised fingers using MediaPipe Hands (now supports TWO hands: 0–10 total)
 - Speaks a predefined Tamil message (gTTS + pygame)
 - Optionally sends an SMS via Twilio (when configured)

Recent improvements:
 - Environment-based Twilio config + runtime toggle
 - Safe audio playback (no busy-spin)
 - Handedness-aware per-hand counting (thumb logic) now summed across both hands (e.g., 4 left + 3 right => 7)
 - Stabilization with majority vote buffer (customizable length)
 - Rate limiting (per gesture) & duplication suppression
 - On-screen overlay with live status & messages
 - Keyboard controls (toggle SMS/Voice, buffer size, reset, help)
 - Optional external messages file (messages.json) to customize phrases for counts 0–10

Keyboard shortcuts:
 q: Quit | s: Toggle SMS | v: Toggle Voice | r: Reset memory | +/-: Buffer size | h: Help

Customization:
 Create a messages.json with a mapping of stringified counts to phrases, e.g.:
 {
     "1": "Food please", "6": "Call my family", "10": "Emergency!"
 }
 Missing keys fall back to defaults.
"""

# === New Extended Features ===
# - config.json (runtime tunables: buffer_size, cooldown_seconds, stable_ratio, global_min_interval, beep_enabled, logging_enabled)
# - Enhanced stability: majority ratio + median agreement
# - Global minimum interval between ANY triggers
# - Event logging (CSV) with timestamp and actions taken
# - Optional beep sound (pure tone) on detection (toggle 'b')
# - Show per-hand counts on overlay
# - Hotkeys: b (beep on/off), l (logging on/off), c (reload config)

# -----------------------------
# Twilio Setup (from environment)
# -----------------------------
# Set these environment variables or hardcode cautiously for local testing:
#   TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, DEST_PHONE_NUMBER
account_sid = os.getenv('TWILIO_ACCOUNT_SID', '').strip()
auth_token = os.getenv('TWILIO_AUTH_TOKEN', '').strip()
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER', '').strip()  # e.g., +1234567890
destination_phone_number = os.getenv('DEST_PHONE_NUMBER', '').strip()

twilio_client = None
SMS_ENABLED = False
if account_sid and auth_token and twilio_phone_number and destination_phone_number:
    try:
        twilio_client = Client(account_sid, auth_token)
        SMS_ENABLED = True
    except Exception as e:
        print(f"[Twilio] Failed to initialize client: {e}")
        SMS_ENABLED = False
else:
    print("[Twilio] Not configured (set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, DEST_PHONE_NUMBER). SMS disabled.")

# -----------------------------
# Initialize pygame for audio
# -----------------------------
AUDIO_ENABLED = True
if pygame is None:
    AUDIO_ENABLED = False
else:
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"[Audio] pygame.mixer.init() failed: {e}. Voice disabled.")
        AUDIO_ENABLED = False

def _is_streamlit_runtime() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

def _tts_to_file(message: str) -> str:
    tts = gTTS(text=message, lang='ta')
    tmp_dir = Path(tempfile.gettempdir())
    out_path = tmp_dir / f"emergency_hg_tts_{int(time.time() * 1000)}.mp3"
    tts.save(str(out_path))
    return str(out_path)

def speak_message_tamil(message: str) -> bool:
    """Synthesize and play a Tamil message using gTTS and pygame.
    Returns True on success, False otherwise.
    """
    try:
        out_path = _tts_to_file(message)
        if _is_streamlit_runtime() and st is not None:
            with open(out_path, 'rb') as audio_file:
                st.audio(audio_file.read(), format='audio/mp3')
            try:
                os.remove(out_path)
            except Exception:
                pass
            return True

        if not AUDIO_ENABLED or pygame is None:
            print(f"[Audio] Saved TTS to {out_path} (playback disabled)")
            return True

        pygame.mixer.music.load(out_path)
        pygame.mixer.music.play()
        # Avoid CPU spin
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        try:
            os.remove(out_path)
        except Exception:
            # On Windows, ensure file is not locked
            pass
        return True
    except Exception as e:
        print(f"[Audio] Failed to speak message: {e}")
        return False

# -----------------------------
# MediaPipe / camera initialization helpers
# -----------------------------
def initialize_hand_pipeline():
    """Create MediaPipe hand detection objects lazily.

    Streamlit Cloud can import this module without opening a webcam or touching
    MediaPipe internals until the app actually needs them.
    """
    try:
        mp_solutions = getattr(mp, "solutions", None)
        if mp_solutions is None:
            from mediapipe import solutions as mp_solutions

        mp_hands = mp_solutions.hands
        mp_draw = mp_solutions.drawing_utils
        hands = mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2,
        )
        return mp_hands, mp_draw, hands
    except Exception as e:
        print(f"[MediaPipe] Failed to initialize hand pipeline: {e}")
        return None, None, None


def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] Failed to open default camera (index 0). Exiting.")
        return None
    return cap

# -----------------------------
# Gesture Buffer for Stability
# -----------------------------
gesture_buffer = collections.deque(maxlen=5)
buffer_size = 5
last_sent_gesture = None  # Last total-finger gesture acted upon
# Initialize cooldown tracking for 0–10
last_sent_time = {i: 0.0 for i in range(0, 11)}
COOLDOWN_SECONDS = 30  # Minimum time between repeated actions for the same gesture
GLOBAL_MIN_INTERVAL = 3.0  # Minimum interval between any two triggers
last_any_trigger_time = 0.0
STABLE_RATIO = 0.8  # Majority ratio threshold (can be overridden by config)
LOGGING_ENABLED = True
BEEP_ENABLED = False
CONFIG_PATH = "config.json"
EVENT_LOG_PATH = "event_log.csv"
HIGH_CONTRAST = False  # High contrast / large text mode
INACTIVITY_WARNING_SECONDS = 15  # Show gentle reminder after this idle period (no hand)
INACTIVITY_ALERT_SECONDS = 60    # Escalate notification overlay after prolonged inactivity
DWELL_WINDOWS = 1  # Number of consecutive stable windows required to fire an action
stable_candidate = None  # Tracks current gesture under dwell evaluation
stable_candidate_count = 0  # Number of consecutive stable windows for candidate
last_activity_time = time.time()

# Severity mapping for color codes / escalation
SEVERITY = {
    0: "none",
    1: "need_food",
    2: "restroom",
    3: "water",
    4: "pain",
    5: "help",
    6: "family",
    7: "doctor",
    8: "medicine",
    9: "rest",
    10: "emergency",
}

# Runtime toggles
VOICE_ENABLED = True

def set_buffer_size(n: int):
    global gesture_buffer, buffer_size
    n = max(3, min(15, int(n)))
    buffer_size = n
    gesture_buffer = collections.deque(list(gesture_buffer), maxlen=buffer_size)

def show_help():
    print("\nControls:\n q: Quit\n s: Toggle SMS on/off\n v: Toggle Voice on/off\n r: Reset last sent memory\n +/-: Change buffer size\n h: Help\n")
    print(" b: Toggle beep on detection\n l: Toggle event logging\n c: Reload config.json\n")
    print(" k: Toggle high contrast mode\n d: Increase dwell windows (Shift+d decreases)\n")

DEFAULT_MESSAGES = {
    0: "",  # No fingers / unclear
    1: "எனக்கு உணவு வேண்டும்!",              # I want food
    2: "கழிவறைக்குச் செல்ல வேண்டும்!",        # Need restroom
    3: "எனக்கு தண்ணீர் வேண்டும்!",          # I want water
    4: "எனக்கு வலி உள்ளது!",                 # I am in pain
    5: "உங்கள் உதவி தேவை!",                  # Need your help
    6: "தயவு செய்து குடும்பத்தினரை அழைக்கவும்!",  # Call my family
    7: "மருத்துவரை அழைக்கவும்!",              # Call the doctor
    8: "எனக்கு மருந்து வேண்டும்!",            # I need medicine
    9: "சிறிது ஓய்வு வேண்டும்!",             # I need some rest
    10: "அவசரம்! உடனடி உதவி தேவை!",         # Emergency
}

def load_messages():
    path = "messages.json"
    if not os.path.isfile(path):
        return DEFAULT_MESSAGES.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert string keys to ints, merge onto defaults
        merged = DEFAULT_MESSAGES.copy()
        for k, v in data.items():
            try:
                ki = int(k)
                if 0 <= ki <= 10:
                    merged[ki] = str(v)
            except ValueError:
                continue
        return merged
    except Exception as e:
        print(f"[Messages] Failed to load messages.json: {e}. Using defaults.")
        return DEFAULT_MESSAGES.copy()

messages_tamil = load_messages()

def load_config():
    global buffer_size, COOLDOWN_SECONDS, STABLE_RATIO, GLOBAL_MIN_INTERVAL, BEEP_ENABLED, LOGGING_ENABLED, HIGH_CONTRAST, INACTIVITY_WARNING_SECONDS, INACTIVITY_ALERT_SECONDS, DWELL_WINDOWS
    if not os.path.isfile(CONFIG_PATH):
        return False
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        if 'buffer_size' in cfg:
            set_buffer_size(cfg['buffer_size'])
        if 'cooldown_seconds' in cfg:
            COOLDOWN_SECONDS = float(cfg['cooldown_seconds'])
        if 'stable_ratio' in cfg:
            STABLE_RATIO = max(0.5, min(1.0, float(cfg['stable_ratio'])))
        if 'global_min_interval' in cfg:
            GLOBAL_MIN_INTERVAL = max(0.0, float(cfg['global_min_interval']))
        if 'beep_enabled' in cfg:
            BEEP_ENABLED = bool(cfg['beep_enabled'])
        if 'logging_enabled' in cfg:
            LOGGING_ENABLED = bool(cfg['logging_enabled'])
        if 'high_contrast' in cfg:
            HIGH_CONTRAST = bool(cfg['high_contrast'])
        if 'inactivity_warning_seconds' in cfg:
            INACTIVITY_WARNING_SECONDS = max(5, float(cfg['inactivity_warning_seconds']))
        if 'inactivity_alert_seconds' in cfg:
            INACTIVITY_ALERT_SECONDS = max(INACTIVITY_WARNING_SECONDS + 5, float(cfg['inactivity_alert_seconds']))
        if 'dwell_windows' in cfg:
            DWELL_WINDOWS = max(1, int(cfg['dwell_windows']))
        print(f"[Config] Loaded {CONFIG_PATH}")
        return True
    except Exception as e:
        print(f"[Config] Failed to load {CONFIG_PATH}: {e}")
        return False

def ensure_event_log_header():
    if not LOGGING_ENABLED:
        return
    if not os.path.isfile(EVENT_LOG_PATH):
        try:
            with open(EVENT_LOG_PATH, 'w', encoding='utf-8') as f:
                f.write("timestamp,gesture,total_fingers,message,sms_sent,voice_played\n")
        except Exception as e:
            print(f"[Log] Cannot write header: {e}")

def log_event(gesture:int, message:str, sms_sent:bool, voice_played:bool):
    if not LOGGING_ENABLED:
        return
    try:
        with open(EVENT_LOG_PATH, 'a', encoding='utf-8') as f:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            safe_msg = message.replace('"', "'").replace('\n', ' ').strip()
            line = f"{ts},{gesture},{gesture},\"{safe_msg}\",{int(sms_sent)},{int(voice_played)}\n"
            f.write(line)
    except Exception as e:
        print(f"[Log] Failed to write event: {e}")

def generate_beep(duration=0.15, freq=880):
    if not (AUDIO_ENABLED and BEEP_ENABLED):
        return
    if pygame is None:
        return
    try:
        import math, array
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        buf = array.array('h')
        amplitude = 12000
        for i in range(n_samples):
            t = i / sample_rate
            sample = int(amplitude * math.sin(2 * math.pi * freq * t))
            buf.append(sample)
        snd = pygame.mixer.Sound(buffer=buf.tobytes())
        snd.play()
    except Exception:
        pass

def count_fingers(hand_landmarks, handedness_label: str) -> int:
    """Count raised fingers (0-5) for a single hand using landmarks and handedness.
    - Thumb: compare x positions depending on left/right
    - Other fingers: tip.y < pip.y indicates raised (image coords origin at top-left)
    """
    lm = hand_landmarks.landmark

    # Thumb (tip 4 vs IP 3)
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    is_right = handedness_label.lower().startswith('right')
    thumb_raised = (thumb_tip.x < thumb_ip.x) if is_right else (thumb_tip.x > thumb_ip.x)

    # Other fingers (tip vs pip = tip index, pip = tip-2)
    finger_tips = [8, 12, 16, 20]
    raised = sum(1 for i in finger_tips if lm[i].y < lm[i - 2].y)
    return int(thumb_raised) + raised

def majority_vote(buf: collections.deque) -> int:
    if not buf:
        return 0
    counts = {}
    for v in buf:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)

def median_value(buf: collections.deque) -> int:
    if not buf:
        return 0
    s = sorted(buf)
    mid = len(s)//2
    if len(s) % 2 == 1:
        return s[mid]
    else:
        # even length: pick lower mid to keep integer stable
        return s[mid-1]

def is_stable(buf: collections.deque, candidate: int) -> bool:
    if len(buf) < buffer_size:
        return False
    count = buf.count(candidate)
    ratio = count / len(buf)
    med = median_value(buf)
    return ratio >= STABLE_RATIO and med == candidate

def process_frame(frame, hands, mp_hands, mp_draw):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    per_frame_count = 0
    counts = []

    if results.multi_hand_landmarks:
        handedness_list = []
        if getattr(results, 'multi_handedness', None):
            handedness_list = [h.classification[0].label for h in results.multi_handedness]

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = handedness_list[idx] if idx < len(handedness_list) else 'Right'
            cnt = max(0, min(5, count_fingers(hand_landmarks, label)))
            counts.append(cnt)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        per_frame_count = min(10, sum(counts)) if counts else 0

    return frame, per_frame_count, counts, results

def run_desktop_app():
    global last_activity_time, stable_candidate, stable_candidate_count, last_sent_gesture, last_any_trigger_time, VOICE_ENABLED, SMS_ENABLED, BEEP_ENABLED, LOGGING_ENABLED, HIGH_CONTRAST, DWELL_WINDOWS

    mp_hands, mp_draw, hands = initialize_hand_pipeline()
    if hands is None or mp_hands is None or mp_draw is None:
        raise SystemExit(1)

    cap = initialize_camera()
    if cap is None:
        raise SystemExit(1)

    print("[Init] SMS enabled:", SMS_ENABLED)
    print("[Init] Voice enabled:", AUDIO_ENABLED)
    loaded_cfg = load_config()
    if not loaded_cfg:
        print("[Config] Using defaults (no config.json).")
    ensure_event_log_header()
    show_help()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Frame grab failed. Exiting loop.")
                break

            frame = cv2.flip(frame, 1)
            frame, per_frame_count, counts, results = process_frame(frame, hands, mp_hands, mp_draw)

            if results.multi_hand_landmarks:
                last_activity_time = time.time()

            gesture_buffer.append(per_frame_count)
            mode_count = majority_vote(gesture_buffer)
            stable = is_stable(gesture_buffer, mode_count)

            overlay_msg = messages_tamil.get(mode_count, "")
            per_hand_display = ''
            if counts:
                per_hand_display = ' | Hands:' + '+'.join(str(c) for c in counts)
            status_line = (
                f"Total:{mode_count} | Stable:{'Y' if stable else 'N'} | Dwell:{DWELL_WINDOWS} | SMS:{'Y' if SMS_ENABLED else 'N'} | "
                f"Voice:{'Y' if VOICE_ENABLED and AUDIO_ENABLED else 'N'} | Beep:{'Y' if BEEP_ENABLED else 'N'} | Buf:{buffer_size}" + per_hand_display
            )

            sev = SEVERITY.get(mode_count, "none")
            if sev == "emergency":
                base_color = (0, 0, 255)
            elif sev in ("doctor", "pain"):
                base_color = (0, 140, 255)
            elif sev in ("help", "family", "medicine"):
                base_color = (0, 255, 255)
            else:
                base_color = (0, 255, 0)

            text_color = (255, 255, 255) if HIGH_CONTRAST else base_color
            thickness = 2 if HIGH_CONTRAST else 1

            cv2.putText(frame, status_line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6 if not HIGH_CONTRAST else 0.75, text_color, thickness + 1)
            if overlay_msg:
                cv2.putText(frame, overlay_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75 if not HIGH_CONTRAST else 0.9, text_color, thickness + 1)

            idle_time = time.time() - last_activity_time
            if idle_time >= INACTIVITY_WARNING_SECONDS:
                warn_msg = "கையை கேமராவிற்கு முன்னர் வைத்திருங்கள்"
                if idle_time >= INACTIVITY_ALERT_SECONDS:
                    warn_msg = "உதவி தேவைதா? கை காணப்படவில்லை"
                cv2.putText(frame, warn_msg, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6 if not HIGH_CONTRAST else 0.75, (0, 0, 255) if idle_time >= INACTIVITY_ALERT_SECONDS else (0, 165, 255), 2)

            if stable:
                if mode_count == stable_candidate:
                    pass
                else:
                    stable_candidate = mode_count
                    stable_candidate_count = 0
                stable_candidate_count += 1
            else:
                if mode_count != stable_candidate:
                    stable_candidate = None
                    stable_candidate_count = 0

            dwell_met = stable and stable_candidate == mode_count and stable_candidate_count >= DWELL_WINDOWS

            if dwell_met and mode_count == 10 and int(time.time() * 4) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)

            now = time.time()
            if dwell_met and mode_count in messages_tamil and overlay_msg:
                cooldown_ok = (now - last_sent_time.get(mode_count, 0.0)) >= COOLDOWN_SECONDS
                global_interval_ok = (now - last_any_trigger_time) >= GLOBAL_MIN_INTERVAL
                not_repeated = (mode_count != last_sent_gesture)
                if (cooldown_ok and global_interval_ok) or not_repeated:
                    message = messages_tamil[mode_count]
                    print(f"[Gesture] Detected {mode_count} -> {message}")
                    voice_played = False
                    sms_sent = False
                    if VOICE_ENABLED:
                        if speak_message_tamil(message):
                            voice_played = True
                    if BEEP_ENABLED:
                        generate_beep()
                    if SMS_ENABLED and twilio_client is not None:
                        try:
                            twilio_client.messages.create(body=message, from_=twilio_phone_number, to=destination_phone_number)
                            print("[Twilio] SMS sent.")
                            sms_sent = True
                        except Exception as e:
                            print(f"[Twilio] Failed to send SMS: {e}")
                    log_event(mode_count, message, sms_sent, voice_played)
                    last_sent_gesture = mode_count
                    last_sent_time[mode_count] = now
                    last_any_trigger_time = now

            cv2.imshow("கை சைகை கண்டறிதல்", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                SMS_ENABLED = not SMS_ENABLED
                print(f"[Toggle] SMS {'enabled' if SMS_ENABLED else 'disabled' }.")
            elif key == ord('v'):
                VOICE_ENABLED = not VOICE_ENABLED
                print(f"[Toggle] Voice {'enabled' if VOICE_ENABLED else 'disabled'}.")
            elif key == ord('r'):
                last_sent_gesture = None
                for k in last_sent_time:
                    last_sent_time[k] = 0.0
                print("[Reset] Last sent gesture and timers cleared.")
            elif key == ord('+') or key == ord('='):
                set_buffer_size(buffer_size + 1)
                print(f"[Buffer] Size -> {buffer_size}")
            elif key == ord('-') or key == ord('_'):
                set_buffer_size(buffer_size - 1)
                print(f"[Buffer] Size -> {buffer_size}")
            elif key == ord('h'):
                show_help()
            elif key == ord('b'):
                BEEP_ENABLED = not BEEP_ENABLED
                print(f"[Toggle] Beep {'enabled' if BEEP_ENABLED else 'disabled'}.")
            elif key == ord('l'):
                LOGGING_ENABLED = not LOGGING_ENABLED
                print(f"[Toggle] Logging {'enabled' if LOGGING_ENABLED else 'disabled'}.")
                if LOGGING_ENABLED:
                    ensure_event_log_header()
            elif key == ord('c'):
                load_config()
            elif key == ord('k'):
                HIGH_CONTRAST = not HIGH_CONTRAST
                print(f"[Toggle] High contrast {'enabled' if HIGH_CONTRAST else 'disabled'}.")
            elif key == ord('d'):
                DWELL_WINDOWS += 1
                print(f"[Dwell] Windows required -> {DWELL_WINDOWS}")
            elif key == ord('D'):
                DWELL_WINDOWS = max(1, DWELL_WINDOWS - 1)
                print(f"[Dwell] Windows required -> {DWELL_WINDOWS}")
    finally:
        try:
            hands.close()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if AUDIO_ENABLED and pygame is not None:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

def run_streamlit_app():
    if st is None:
        raise RuntimeError("Streamlit is not installed.")

    mp_hands, mp_draw, hands = initialize_hand_pipeline()
    if hands is None or mp_hands is None or mp_draw is None:
        st.error("MediaPipe could not be initialized in this environment.")
        st.stop()

    st.set_page_config(page_title="Emergency Hand Gesture Assistive Tool", layout="wide")
    st.title("Emergency Hand Gesture Assistive Tool")
    st.write("Deployable demo mode for Streamlit Cloud. Upload an image or take a camera snapshot to detect raised fingers.")
    st.info("Live local webcam capture is only supported when running the desktop app with `python main.py`.")

    uploaded = st.camera_input("Take a snapshot")
    file_upload = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

    image_bytes = None
    if file_upload is not None:
        image_bytes = file_upload.getvalue()
    elif uploaded is not None:
        image_bytes = uploaded.getvalue()

    if image_bytes is None:
        st.stop()

    import numpy as np
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not decode the image.")
        st.stop()

    frame = cv2.flip(frame, 1)
    annotated, per_frame_count, counts, results = process_frame(frame, hands, mp_hands, mp_draw)
    mode_count = max(0, min(10, per_frame_count))
    message = messages_tamil.get(mode_count, "")

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detected fingers: {mode_count}", use_container_width=True)
    with col2:
        st.metric("Total fingers", mode_count)
        st.write("Per-hand counts:", counts if counts else "No hand detected")
        st.write("Tamil message:", message if message else "No mapped message")

        if message:
            if st.button("Play voice output"):
                speak_message_tamil(message)

    try:
        hands.close()
    except Exception:
        pass

def main():
    if _is_streamlit_runtime():
        run_streamlit_app()
    else:
        run_desktop_app()

if __name__ == "__main__":
    main()