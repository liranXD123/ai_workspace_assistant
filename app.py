import os
import time
import av
import cv2
import numpy as np
import bcrypt
import streamlit as st
from ultralytics import YOLO
from plyer import notification
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from langchain_core.prompts import PromptTemplate

# Import LLM Backends
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from langchain_ollama import OllamaLLM

# Import Database Module
import database as db

# ------------------------------------------------------------------------------
# 1. Page Configuration & Database Initialization
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Workspace Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize database schema on startup
db.init_db()

# WebRTC STUN Configuration for Cloud NAT Traversal
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ------------------------------------------------------------------------------
# 2. Model Loading (Cached)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_yolo_models():
    pose = YOLO('yolov8n-pose.pt')
    detect = YOLO('yolov8n.pt')
    return pose, detect


pose_model, detect_model = load_yolo_models()

POSTURE_SENSITIVITY = 0.12
NOTIFICATION_COOLDOWN = 10

# ------------------------------------------------------------------------------
# 3. Session State Initialization
# ------------------------------------------------------------------------------
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

if "is_calibrated" not in st.session_state:
    st.session_state.is_calibrated = False
    st.session_state.base_nose_y = 0
    st.session_state.base_shoulder_y = 0
    st.session_state.base_shoulder_width = 0

if "is_on_break" not in st.session_state:
    st.session_state.is_on_break = False

if "total_bad_posture" not in st.session_state:
    st.session_state.total_bad_posture = 0.0
    st.session_state.total_phone = 0.0
    st.session_state.total_looking_away = 0.0
    st.session_state.total_away = 0.0

if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

if "request_calibration" not in st.session_state:
    st.session_state.request_calibration = False


# ------------------------------------------------------------------------------
# 4. Helper Functions
# ------------------------------------------------------------------------------
def send_os_notification(title, message):
    try:
        notification.notify(
            title=title,
            message=message,
            app_name='AI Workspace Assistant',
            timeout=3
        )
    except Exception as e:
        print(f"[Notification Error] {e}")


def generate_end_of_session_report():
    template = """
    You are an expert AI Ergonomics & Productivity Coach.
    A work session has just been completed. Here are the final tracked metrics:

    - Total Bad Posture Duration: {bad_posture_seconds} seconds
    - Total Phone Usage Duration: {phone_seconds} seconds
    - Total Time Looking Away From Screen: {looking_away_seconds} seconds
    - Total Time Away From Desk: {away_from_desk_seconds} seconds

    Please provide a concise, structured End-of-Session Executive Report in English:
    1. Overall Productivity & Posture Score (out of 100).
    2. Key Observations (what went well vs. main distractions/slouching).
    3. Actionable Advice for the next session.
    """
    prompt = PromptTemplate(
        input_variables=["bad_posture_seconds", "phone_seconds", "looking_away_seconds", "away_from_desk_seconds"],
        template=template
    )

    formatted_prompt = prompt.format(
        bad_posture_seconds=int(st.session_state.total_bad_posture),
        phone_seconds=int(st.session_state.total_phone),
        looking_away_seconds=int(st.session_state.total_looking_away),
        away_from_desk_seconds=int(st.session_state.total_away)
    )

    try:
        # Check if Groq API Key is configured for Cloud execution
        groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

        if groq_api_key and ChatGroq is not None:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.2-3b-preview"
            )
            response = llm.invoke(formatted_prompt)
            report = response.content
        else:
            # Fallback to local Ollama execution
            ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            llm = OllamaLLM(
                model="llama3.2:3b",
                base_url=ollama_base_url
            )
            report = llm.invoke(formatted_prompt)

        st.session_state.ai_report = report

        # Auto-save session metrics & report to SQLite database
        if st.session_state.logged_in_user:
            db.save_session(
                user_id=st.session_state.logged_in_user["id"],
                bad_posture=int(st.session_state.total_bad_posture),
                phone=int(st.session_state.total_phone),
                looking_away=int(st.session_state.total_looking_away),
                away=int(st.session_state.total_away),
                ai_report=report
            )
    except Exception as e:
        st.session_state.ai_report = f"❌ Error generating AI report: {e}"


# ------------------------------------------------------------------------------
# 5. WebRTC Video Frame Processor Class
# ------------------------------------------------------------------------------
class PostureVideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Pose Estimation
        pose_results = pose_model(img, verbose=False)[0]
        annotated_frame = pose_results.plot()

        person_detected = False
        phone_detected = False

        if pose_results.keypoints is not None and len(pose_results.keypoints.data) > 0:
            kpts = pose_results.keypoints.data[0].cpu().numpy()

            if len(kpts) >= 7 and kpts[5][2] > 0.4 and kpts[6][2] > 0.4:
                person_detected = True
                nose_y = kpts[0][1]
                l_sh, r_sh = kpts[5][:2], kpts[6][:2]

                curr_shoulder_y = (l_sh[1] + r_sh[1]) / 2
                curr_shoulder_width = np.linalg.norm(l_sh - r_sh)

                # Process Calibration Request
                if st.session_state.get("request_calibration", False):
                    st.session_state.base_nose_y = nose_y
                    st.session_state.base_shoulder_y = curr_shoulder_y
                    st.session_state.base_shoulder_width = curr_shoulder_width
                    st.session_state.is_calibrated = True
                    st.session_state.request_calibration = False

                # Evaluate Posture
                if st.session_state.get("is_calibrated", False):
                    b_width = st.session_state.base_shoulder_width
                    nose_drop = (nose_y - st.session_state.base_nose_y) / b_width
                    shoulder_drop = (curr_shoulder_y - st.session_state.base_shoulder_y) / b_width

                    if nose_drop > POSTURE_SENSITIVITY or shoulder_drop > POSTURE_SENSITIVITY:
                        if not st.session_state.get("is_on_break", False):
                            st.session_state.total_bad_posture += 0.1

        # Object Detection (Phone)
        detect_results = detect_model(img, verbose=False)[0]
        for box in detect_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 67 and conf > 0.4:
                phone_detected = True
                if not st.session_state.get("is_on_break", False):
                    st.session_state.total_phone += 0.1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (50, 50, 255), 2)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# ------------------------------------------------------------------------------
# 6. Header & Authentication
# ------------------------------------------------------------------------------
st.title("🤖 AI Workspace & Ergonomics Assistant")
st.markdown("Real-time Computer Vision & LLM Productivity Dashboard")
st.divider()

# --- LOGIN / SIGNUP PORTAL ---
if not st.session_state.logged_in_user:
    st.subheader("🔒 Authentication Required")
    auth_tab_login, auth_tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])

    with auth_tab_login:
        st.markdown("##### Log in to access your dashboard")
        login_user = st.text_input("Username", key="login_user_input")
        login_pass = st.text_input("Password", type="password", key="login_pass_input")

        if st.button("Log In", type="primary"):
            if login_user and login_pass:
                user_record = db.get_user_by_username(login_user)
                if user_record:
                    user_id, uname, stored_hash, _ = user_record
                    if bcrypt.checkpw(login_pass.encode('utf-8'), stored_hash.encode('utf-8')):
                        st.session_state.logged_in_user = {"id": user_id, "username": uname}
                        st.success(f"Welcome back, {uname}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.error("Invalid username or password.")
            else:
                st.warning("Please enter both username and password.")

    with auth_tab_signup:
        st.markdown("##### Create a new account")
        signup_user = st.text_input("Choose Username", key="signup_user_input")
        signup_pass = st.text_input("Choose Password", type="password", key="signup_pass_input")
        signup_pass_confirm = st.text_input("Confirm Password", type="password", key="signup_pass_confirm_input")

        if st.button("Sign Up"):
            if signup_user and signup_pass:
                if signup_pass != signup_pass_confirm:
                    st.error("Passwords do not match!")
                elif len(signup_pass) < 6:
                    st.warning("Password must be at least 6 characters long.")
                else:
                    hashed_bytes = bcrypt.hashpw(signup_pass.encode('utf-8'), bcrypt.gensalt())
                    hashed_str = hashed_bytes.decode('utf-8')

                    success = db.create_user(signup_user, hashed_str)
                    if success:
                        st.success("Account created successfully! You can now log in.")
                    else:
                        st.error("Username already taken. Please choose another one.")
            else:
                st.warning("Please fill in all fields.")

    st.stop()

# ------------------------------------------------------------------------------
# 7. Main Dashboard View (For Authenticated Users)
# ------------------------------------------------------------------------------
col_user_info, col_logout = st.columns([4, 1])
with col_user_info:
    st.success(f"👤 Logged in as: **{st.session_state.logged_in_user['username']}**")
with col_logout:
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.logged_in_user = None
        st.rerun()

main_tab_tracker, main_tab_history = st.tabs(["📹 Live Workspace Tracker", "📊 Session History"])

# ------------------------------------------------------------------------------
# TAB 1: Live Workspace Tracker
# ------------------------------------------------------------------------------
with main_tab_tracker:
    col_video, col_metrics = st.columns([2, 1])

    with col_video:
        st.subheader("📹 Live Browser Stream")

        # WebRTC Browser Camera Streamer
        webrtc_ctx = webrtc_streamer(
            key="workspace-assistant-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=PostureVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("##### 🎛️ Session Controls")
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button("🎯 Calibrate Posture", use_container_width=True):
                st.session_state.request_calibration = True
                st.toast("Calibration requested!")

        with btn_col2:
            break_label = "💪 Resume Work" if st.session_state.is_on_break else "☕ Take Break"
            if st.button(break_label, use_container_width=True):
                st.session_state.is_on_break = not st.session_state.is_on_break
                st.rerun()

        with btn_col3:
            if st.button("📋 Generate Report", use_container_width=True, type="primary"):
                with st.spinner("Generating AI Executive Summary..."):
                    generate_end_of_session_report()
                st.rerun()

    with col_metrics:
        st.subheader("📊 Session Analytics")

        if st.session_state.is_on_break:
            st.warning("☕ Status: On Break (Metrics Paused)")
        elif not st.session_state.is_calibrated:
            st.error("⚠️ Status: Active - Calibration Needed!")
        else:
            st.success("🟢 Status: Active & Tracking")

        st.markdown("---")
        st.metric("Bad Posture Time", f"{int(st.session_state.total_bad_posture)}s")
        st.metric("Phone Usage Time", f"{int(st.session_state.total_phone)}s")

    # Render AI Report below tracker
    if st.session_state.ai_report:
        st.divider()
        st.success("✅ **Session saved!** Click on the **📊 Session History** tab above to view all past records.")
        st.subheader("📊 Latest AI Summary Report")
        st.info(st.session_state.ai_report)

# ------------------------------------------------------------------------------
# TAB 2: Session History
# ------------------------------------------------------------------------------
with main_tab_history:
    st.subheader("📜 Your Past Work Sessions")
    user_sessions = db.get_user_sessions(st.session_state.logged_in_user["id"])

    if not user_sessions:
        st.info("No recorded sessions found yet. Start a session in the live tracker tab!")
    else:
        for idx, sess in enumerate(user_sessions):
            sess_id, timestamp, bad_posture, phone, looking_away, away, report = sess

            is_latest = (idx == 0)
            label_prefix = "⭐ [LATEST SESSION]" if is_latest else "🗓️"

            with st.expander(f"{label_prefix} Session #{sess_id} — {timestamp}", expanded=is_latest):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Bad Posture", f"{bad_posture}s")
                c2.metric("Phone Time", f"{phone}s")
                c3.metric("Looking Away", f"{looking_away}s")
                c4.metric("Away From Desk", f"{away}s")

                st.markdown("---")
                st.markdown("**🤖 AI Executive Summary:**")
                st.write(report if report else "No AI report generated for this session.")