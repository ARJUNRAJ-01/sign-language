import cv2
import time
import streamlit as st
from PIL import Image
from src.inference import SignLanguageInference
from src.smoothing import TemporalSmoother
from src.word_builder import WordBuilder

# Must be the first Streamlit command
st.set_page_config(
    page_title="SignBridge - ASL Recognition",
    page_icon="ðŸ–ï¸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title { font-size: 2.8rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
    .subtitle { font-size: 1.15rem; color: #4B5563; margin-bottom: 16px; }
    .prediction-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .letter-display { font-size: 4.5rem; font-weight: 900; color: #2563EB; line-height: 1.1; }
    .word-display { font-size: 2.5rem; font-weight: 800; color: #059669; letter-spacing: 0.08em; min-height: 60px; }
    .status-text { font-size: 1.1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'word_builder' not in st.session_state:
    st.session_state.word_builder = WordBuilder(cooldown=1.2)
if 'smoother' not in st.session_state:
    st.session_state.smoother = TemporalSmoother(window_size=5)
if 'inference' not in st.session_state:
    try:
        st.session_state.inference = SignLanguageInference()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'auto_add' not in st.session_state:
    st.session_state.auto_add = True
if 'last_auto_added_prediction' not in st.session_state:
    st.session_state.last_auto_added_prediction = None
if 'camera' not in st.session_state:
    st.session_state.camera = None
    
# Callback functions for buttons to prevent stream interrupts
def btn_add_letter():
    pred = st.session_state.get("current_prediction")
    if pred and pred not in ["No hand detected", "Invalid crop", "-"]:
        st.session_state.word_builder.add_letter(pred)
    st.rerun()

def btn_space():
    st.session_state.word_builder.add_letter("space")
    st.rerun()

def btn_delete():
    st.session_state.word_builder.delete_last()
    st.rerun()

def btn_clear():
    st.session_state.word_builder.clear()
    st.rerun()
# UI Header
st.markdown('<p class="main-title">SignBridge</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-Time Sign Language Alphabet Recognition & Sentence Construction</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1.2])

with col1:
    st.markdown("### Live Camera Feed")
    frame_placeholder = st.empty()
    
    col_start, col_stop, col_auto = st.columns([1, 1, 1.4])
    with col_start:
        if st.button("Start Camera", key="start_cam", use_container_width=True):
            st.session_state.run_webcam = True
    with col_stop:
        if st.button("Stop Camera", key="stop_cam", use_container_width=True):
            st.session_state.run_webcam = False
            if st.session_state.camera is not None:
                st.session_state.camera.release()
                st.session_state.camera = None
    with col_auto:
        st.session_state.auto_add = st.checkbox("Hands-Free Auto Add", value=st.session_state.auto_add, help="Automatically types the letter when held steady.")

with col2:
    st.markdown("### Recognition")
    pred_placeholder = st.empty()

    st.markdown("### Word Builder")
    word_placeholder = st.empty()

    def word_controls():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.button("Add Letter", on_click=btn_add_letter, use_container_width=True)
        with c2:
            st.button("Space", on_click=btn_space, use_container_width=True)
        with c3:
            st.button("Delete", on_click=btn_delete, use_container_width=True)
        with c4:
            st.button("Clear", on_click=btn_clear, use_container_width=True)

    word_controls()

    st.markdown("### Sentence History")
    history_placeholder = st.empty()

# Webcam Loop
if st.session_state.run_webcam:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Camera warmup retry
    if not cap.isOpened():
        time.sleep(0.5)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
    if not cap.isOpened():
        st.error("Cannot access webcam. Make sure no other application is using it.")
        st.session_state.run_webcam = False
        
    while st.session_state.run_webcam:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
            
        frame = cv2.flip(frame, 1) # Mirror for natural feel
        
        # Inference
        try:
            processed_frame, raw_pred, conf, top_candidates = st.session_state.inference.predict(frame)
        except Exception as e:
            st.error(f"Inference error: {e}")
            break
            
        # Temporal Smoothing
        if raw_pred not in ["No hand detected", "Invalid crop"]:
            stable_pred = st.session_state.smoother.update(raw_pred)
        else:
            stable_pred = None
            st.session_state.smoother.clear()
            
        candidate_count = getattr(st.session_state.smoother, "candidate_count", 0)
        display_pred = stable_pred if (stable_pred and candidate_count >= 3) else raw_pred
        st.session_state.current_prediction = display_pred if display_pred not in ["No hand detected", "Invalid crop", "-"] else None
        # Hands-Free Auto Add: require confidence + temporal confirmation
        candidate_count = getattr(st.session_state.smoother, "candidate_count", 0)

        if (
            st.session_state.auto_add
            and stable_pred
            and conf >= 0.40
            and candidate_count >= 3
            and stable_pred != st.session_state.last_auto_added_prediction
        ):
            st.session_state.word_builder.add_letter(stable_pred)
            st.session_state.last_auto_added_prediction = stable_pred
            
        # Determine status text and color
        if raw_pred == "No hand detected":
            # A new gesture after hand removal is a fresh input event.
            st.session_state.last_auto_added_prediction = None
            status_color = "#6B7280"
            status_text = "No Hand Detected"
            display_pred = "-"
            conf = 0.0
        elif conf >= 0.60:
            status_color = "#10B981"
            status_text = "High Confidence"
        elif conf >= 0.30:
            status_color = "#F59E0B"
            status_text = "Active (Moderate Confidence)"
        else:
            status_color = "#EF4444"
            status_text = "Low Certainty"
            
        # Update UI Frame
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
        
        candidates_html = ""
        if top_candidates and len(top_candidates) > 1:
            cand_items = "".join([
                f"<span style='display:inline-block; margin:2px 5px; padding:3px 8px; background:#EEF2F6; border-radius:6px; font-size:0.88rem; font-weight:600;'>{c}: {p*100:.0f}%</span>"
                for c, p in top_candidates
            ])
            candidates_html = f"<div style='margin-top:6px;'><b>Alternatives:</b><br>{cand_items}</div>"

        # Decision-layer diagnostics
        if raw_pred == "No hand detected":
            hand_status = "NOT DETECTED"
        elif raw_pred in ["Invalid crop", "Model not ready"]:
            hand_status = "UNAVAILABLE"
        else:
            hand_status = "GOOD"

        candidate_count = getattr(st.session_state.smoother, "candidate_count", 0)
        stable_state = getattr(st.session_state.smoother, "stable_prediction", None)

        if stable_state and candidate_count >= 3:
            temporal_status = "HIGH"
        elif candidate_count >= 2:
            temporal_status = "BUILDING"
        else:
            temporal_status = "LOW"

        accepted = bool(
            stable_pred
            and conf >= 0.40
            and candidate_count >= 3
            and stable_state == stable_pred
        )
        decision_text = "? ACCEPTED" if accepted else "? WAITING"
        decision_color = "#10B981" if accepted else "#F59E0B"

        pred_html = f"""
        <div class="prediction-box">
            <p style="margin:0; color:#6B7280; font-weight:bold; font-size:0.9rem;">PREDICTION</p>
            <div class="letter-display">{display_pred}</div>
            <p style="margin:0 0 10px 0; font-size:1.25rem; font-weight:bold; color:#374151;">Confidence: {conf*100:.1f}%</p>
            <p style="margin:4px 0; text-align:left; color:#374151;"><b>Prediction:</b> {display_pred}</p>
            <p style="margin:4px 0; text-align:left; color:#374151;"><b>Confidence:</b> {conf*100:.1f}%</p>
            <p style="margin:4px 0; text-align:left; color:#374151;"><b>Temporal Stability:</b> {temporal_status}</p>
            <p style="margin:4px 0; text-align:left; color:#374151;"><b>Hand Detection:</b> {hand_status}</p>
            <p style="margin:10px 0 0 0; font-weight:bold; color:{decision_color};">{decision_text}</p>
            {candidates_html}
        </div>
        """
        pred_placeholder.markdown(pred_html, unsafe_allow_html=True)
        
        current_word = st.session_state.word_builder.get_word()
        word_html = f"""
        <div class="prediction-box" style="margin-top: 10px;">
            <p style="margin:0; color:#6B7280; font-weight:bold; font-size:0.9rem;">CURRENT WORD</p>
            <div class="word-display">{current_word if current_word else "&nbsp;"}</div>
        </div>
        """
        word_placeholder.markdown(word_html, unsafe_allow_html=True)
        
        history_list = ", ".join(st.session_state.word_builder.get_history()[-10:])
        history_placeholder.info(history_list if history_list else "No history yet.")
        
        # Yield execution
        time.sleep(0.01)

    cap.release()
else:
    frame_placeholder.info("Click 'Start Camera' to begin recognition.")
    word_html = f"""
    <div class="prediction-box" style="margin-top: 10px;">
        <p style="margin:0; color:#6B7280; font-weight:bold; font-size:0.9rem;">CURRENT WORD</p>
        <div class="word-display">{st.session_state.word_builder.get_word() if st.session_state.word_builder.get_word() else "&nbsp;"}</div>
    </div>
    """
    word_placeholder.markdown(word_html, unsafe_allow_html=True)
    history_list = ", ".join(st.session_state.word_builder.get_history()[-10:])
    history_placeholder.info(history_list if history_list else "No history yet.")
