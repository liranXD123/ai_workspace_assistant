import cv2
import numpy as np
import time
import os
import bcrypt
import streamlit as st
from ultralytics import YOLO
from plyer import notification
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

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


# ------------------------------------------------------------------------------
# 2. Model & Resource Loading (Cached)
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
    st.session_state.logged_in_user = None  # Holds dict: {"id": int, "username": str}

if "is_calibrated" not in st.session_state:
    st.session_state.is_calibrated = False
    st.session_state.base_nose_y = 0
    st.session_state.base_shoulder_y = 0
    st.session_state.base_shoulder_width = 0

if "is_on_break" not in st.session_state:
    st.session_state.is_on_break = False

if "is_running" not in st.session_state:
    st.session_state.is_running = False

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


def draw_text_clean(img, text, pos, font_scale=0.55, color=(255, 255, 255), thickness=1):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


import os
import streamlit as st
from langchain_core.prompts import PromptTemplate

# Import LLM handlers
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from langchain_ollama import OllamaLLM


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
        # Check if Groq API Key is available in Streamlit Secrets or Environment Variables
        groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

        if groq_api_key and ChatGroq is not None:
            # High-speed cloud LLM execution via Groq
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.2-3b-preview"
            )
            response = llm.invoke(formatted_prompt)
            report = response.content
        else:
            # Fallback to local Ollama on PC/Docker
            ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            llm = OllamaLLM(
                model="llama3.2:3b",
                base_url=ollama_base_url
            )
            report = llm.invoke(formatted_prompt)

        st.session_state.ai_report = report

        # Auto-save session to SQLite DB for logged-in user
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
# 5. Header & Authentication View
# ------------------------------------------------------------------------------
st.title("🤖 AI Workspace & Ergonomics Assistant")
st.markdown("Real-time Computer Vision & LLM Productivity Dashboard")
st.divider()

# --- LOGIN / SIGNUP PORTAL (If user is NOT logged in) ---
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
                    # tuple structure: (ID, username, password_hash, created_at)
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
                    # Hash password using bcrypt before saving
                    hashed_bytes = bcrypt.hashpw(signup_pass.encode('utf-8'), bcrypt.gensalt())
                    hashed_str = hashed_bytes.decode('utf-8')

                    success = db.create_user(signup_user, hashed_str)
                    if success:
                        st.success("Account created successfully! You can now log in.")
                    else:
                        st.error("Username already taken. Please choose another one.")
            else:
                st.warning("Please fill in all fields.")

    st.stop()  # Stop execution until user logs in

# ------------------------------------------------------------------------------
# 6. Main Dashboard View (For Authenticated Users)
# ------------------------------------------------------------------------------
col_user_info, col_logout = st.columns([4, 1])
with col_user_info:
    st.success(f"👤 Logged in as: **{st.session_state.logged_in_user['username']}**")
with col_logout:
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.logged_in_user = None
        st.session_state.is_running = False
        st.rerun()

# Define Main Tabs
main_tab_tracker, main_tab_history = st.tabs(["📹 Live Workspace Tracker", "📊 Session History"])

# ------------------------------------------------------------------------------
# TAB 1: Live Workspace Tracker
# ------------------------------------------------------------------------------
with main_tab_tracker:
    col_video, col_metrics = st.columns([2, 1])

    with col_video:
        st.subheader("📹 Live Feed & Controls")

        # Video Frame Target
        video_placeholder = st.empty()

        # --- IN-LINE CONTROL PANEL ---
        st.markdown("##### 🎛️ Session Controls")
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        with btn_col1:
            if not st.session_state.is_running:
                if st.button("▶️ Start", use_container_width=True, type="primary"):
                    st.session_state.is_running = True
                    st.session_state.ai_report = None
                    st.rerun()
            else:
                if st.button("⏹️ Stop & Report", use_container_width=True, type="primary"):
                    st.session_state.is_running = False
                    with st.spinner("Generating AI Report & saving session to database..."):
                        generate_end_of_session_report()
                    st.rerun()

        with btn_col2:
            if st.button("🎯 Calibrate", use_container_width=True, disabled=not st.session_state.is_running):
                st.session_state.request_calibration = True

        with btn_col3:
            break_label = "💪 Resume" if st.session_state.is_on_break else "☕ Break"
            if st.button(break_label, use_container_width=True, disabled=not st.session_state.is_running):
                st.session_state.is_on_break = not st.session_state.is_on_break
                status = "On Break" if st.session_state.is_on_break else "Working"
                send_os_notification("Session Status", f"Status changed to: {status}")
                st.rerun()

        with btn_col4:
            if st.button("🧹 Reset", use_container_width=True):
                st.session_state.total_bad_posture = 0.0
                st.session_state.total_phone = 0.0
                st.session_state.total_looking_away = 0.0
                st.session_state.total_away = 0.0
                st.session_state.is_calibrated = False
                st.session_state.ai_report = None
                st.rerun()

    with col_metrics:
        st.subheader("📊 Session Analytics")

        # Real-time Status Banner
        if not st.session_state.is_running:
            st.info("🔴 Status: Session Inactive")
        elif st.session_state.is_on_break:
            st.warning("☕ Status: On Break (Metrics Paused)")
        elif not st.session_state.is_calibrated:
            st.error("⚠️ Status: Active - Calibration Needed!")
        else:
            st.success("🟢 Status: Active & Tracking")

        st.markdown("---")
        metric_posture = st.empty()
        metric_phone = st.empty()
        metric_gaze = st.empty()
        metric_away = st.empty()

    # Banner & AI Report display inside Tracker Tab
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
        # Loop through sessions, expand the first (latest) session automatically
        for idx, sess in enumerate(user_sessions):
            sess_id, timestamp, bad_posture, phone, looking_away, away, report = sess

            # expanded=True for index 0 keeps the newest session open by default
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

# ------------------------------------------------------------------------------
# 7. Vision Processing Loop
# ------------------------------------------------------------------------------
if st.session_state.is_running:
    cap = cv2.VideoCapture(0)
    last_frame_time = time.time()
    last_notification_time = 0

    while cap.isOpened() and st.session_state.is_running:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access the webcam.")
            break

        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time

        person_detected = False
        phone_detected = False
        is_bad_posture = False
        is_looking_away = False

        # --- Pose Analysis ---
        pose_results = pose_model(frame, verbose=False)[0]
        annotated_frame = pose_results.plot()

        if pose_results.keypoints is not None and len(pose_results.keypoints.data) > 0:
            kpts = pose_results.keypoints.data[0].cpu().numpy()

            if len(kpts) >= 7 and kpts[5][2] > 0.4 and kpts[6][2] > 0.4:
                person_detected = True
                nose_x, nose_y = kpts[0][:2]
                l_eye, r_eye = kpts[1][:2], kpts[2][:2]
                l_sh, r_sh = kpts[5][:2], kpts[6][:2]

                curr_shoulder_y = (l_sh[1] + r_sh[1]) / 2
                curr_shoulder_width = np.linalg.norm(l_sh - r_sh)

                # Process Calibration
                if st.session_state.request_calibration:
                    st.session_state.base_nose_y = nose_y
                    st.session_state.base_shoulder_y = curr_shoulder_y
                    st.session_state.base_shoulder_width = curr_shoulder_width
                    st.session_state.is_calibrated = True
                    st.session_state.request_calibration = False
                    send_os_notification("Calibration Complete", "Posture baseline set successfully.")

                # Evaluate Posture
                if st.session_state.is_calibrated:
                    b_width = st.session_state.base_shoulder_width
                    nose_drop = (nose_y - st.session_state.base_nose_y) / b_width
                    shoulder_drop = (curr_shoulder_y - st.session_state.base_shoulder_y) / b_width
                    width_change = abs(curr_shoulder_width - b_width) / b_width

                    if nose_drop > POSTURE_SENSITIVITY or shoulder_drop > POSTURE_SENSITIVITY or width_change > (
                            POSTURE_SENSITIVITY * 1.5):
                        is_bad_posture = True
                        if not st.session_state.is_on_break:
                            st.session_state.total_bad_posture += dt

                # Evaluate Gaze
                if kpts[0][2] > 0.4 and kpts[1][2] > 0.4 and kpts[2][2] > 0.4:
                    eyes_center_x = (l_eye[0] + r_eye[0]) / 2
                    eyes_center_y = (l_eye[1] + r_eye[1]) / 2
                    eye_distance = abs(r_eye[0] - l_eye[0])

                    if eye_distance > 0:
                        eye_nose_ratio_y = (nose_y - eyes_center_y) / curr_shoulder_width
                        is_vertical_away = eye_nose_ratio_y < 0.03 or eye_nose_ratio_y > 0.22

                        horizontal_ratio = abs(nose_x - eyes_center_x) / eye_distance
                        is_horizontal_away = horizontal_ratio > 0.35

                        if is_vertical_away or is_horizontal_away:
                            is_looking_away = True
                            if not st.session_state.is_on_break:
                                st.session_state.total_looking_away += dt

        # --- Away Tracking ---
        if not person_detected and not st.session_state.is_on_break:
            st.session_state.total_away += dt

        # --- Phone Detection ---
        detect_results = detect_model(frame, verbose=False)[0]
        for box in detect_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 67 and conf > 0.4:
                phone_detected = True
                if not st.session_state.is_on_break:
                    st.session_state.total_phone += dt
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (50, 50, 255), 2)

        # --- Notifications ---
        if not st.session_state.is_on_break and (current_time - last_notification_time > NOTIFICATION_COOLDOWN):
            if is_bad_posture:
                send_os_notification("Posture Warning", "Bad posture detected! Please sit up straight.")
                last_notification_time = current_time
            elif phone_detected:
                send_os_notification("Distraction Warning", "Phone detected! Put your phone away.")
                last_notification_time = current_time

        # --- Frame Overlay ---
        if st.session_state.is_on_break:
            draw_text_clean(annotated_frame, "STATUS: ON BREAK ☕", (20, 40), font_scale=0.7, color=(255, 165, 0),
                            thickness=2)

        # Render Frame & Metrics
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        metric_posture.metric("Bad Posture Time", f"{int(st.session_state.total_bad_posture)}s")
        metric_phone.metric("Phone Usage Time", f"{int(st.session_state.total_phone)}s")
        metric_gaze.metric("Looking Away Time", f"{int(st.session_state.total_looking_away)}s")
        metric_away.metric("Away From Desk Time", f"{int(st.session_state.total_away)}s")

    cap.release()