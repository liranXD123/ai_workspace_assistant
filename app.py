import cv2
import numpy as np
import time
import json
import os
import streamlit as st
from ultralytics import YOLO
from plyer import notification
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ------------------------------------------------------------------------------
# 1. Page Configuration (Must be the first Streamlit command)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Workspace Assistant",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------------------------------------------------------
# 2. Model & Resource Loading (Cached to avoid reloading on reruns)
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
if "is_calibrated" not in st.session_state:
    st.session_state.is_calibrated = False
    st.session_state.base_nose_y = 0
    st.session_state.base_shoulder_y = 0
    st.session_state.base_shoulder_width = 0

if "is_on_break" not in st.session_state:
    st.session_state.is_on_break = False

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "metrics" not in st.session_state:
    st.session_state.total_bad_posture = 0.0
    st.session_state.total_phone = 0.0
    st.session_state.total_looking_away = 0.0
    st.session_state.total_away = 0.0

if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

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
    try:
        llm = OllamaLLM(model="llama3")
        formatted_prompt = prompt.format(
            bad_posture_seconds=int(st.session_state.total_bad_posture),
            phone_seconds=int(st.session_state.total_phone),
            looking_away_seconds=int(st.session_state.total_looking_away),
            away_from_desk_seconds=int(st.session_state.total_away)
        )
        report = llm.invoke(formatted_prompt)
        st.session_state.ai_report = report
    except Exception as e:
        st.session_state.ai_report = f"❌ Error generating AI report: {e}"

# ------------------------------------------------------------------------------
# 5. UI Layout - Title & Sidebar
# ------------------------------------------------------------------------------
st.title("🤖 AI Workspace & Ergonomics Assistant")
st.markdown("Real-time Computer Vision & LLM Productivity Dashboard")
st.divider()

st.sidebar.header("🕹️ Session Controls")

# Start / Stop Session Button
if not st.session_state.is_running:
    if st.sidebar.button("▶️ Start Session", use_container_width=True):
        st.session_state.is_running = True
        st.session_state.ai_report = None
        st.rerun()
else:
    if st.sidebar.button("⏹️ End Session & Get Report", use_container_width=True):
        st.session_state.is_running = False
        with st.spinner("Generating AI Executive Report via Ollama..."):
            generate_end_of_session_report()
        st.rerun()

# Calibration Button
if st.sidebar.button("🎯 Calibrate Posture", use_container_width=True):
    st.session_state.request_calibration = True

# Break Toggle Button
break_label = "💪 Resume Work" if st.session_state.is_on_break else "☕ Take a Break"
if st.sidebar.button(break_label, use_container_width=True):
    st.session_state.is_on_break = not st.session_state.is_on_break
    status = "On Break" if st.session_state.is_on_break else "Working"
    send_os_notification("Session Status", f"Status changed to: {status}")

# Reset Metrics Button
if st.sidebar.button("🧹 Reset Metrics", use_container_width=True):
    st.session_state.total_bad_posture = 0.0
    st.session_state.total_phone = 0.0
    st.session_state.total_looking_away = 0.0
    st.session_state.total_away = 0.0
    st.session_state.is_calibrated = False
    st.session_state.ai_report = None
    st.sidebar.success("Metrics reset successfully!")

# ------------------------------------------------------------------------------
# 6. Main Dashboard Layout
# ------------------------------------------------------------------------------
col_video, col_metrics = st.columns([2, 1])

with col_video:
    st.subheader("📹 Live Video Stream")
    video_placeholder = st.empty()

with col_metrics:
    st.subheader("📊 Session Analytics")
    metric_posture = st.empty()
    metric_phone = st.empty()
    metric_gaze = st.empty()
    metric_away = st.empty()

# Render AI Summary Report at the bottom if generated
if st.session_state.ai_report:
    st.divider()
    st.subheader("📊 AI Executive Summary Report")
    st.info(st.session_state.ai_report)

# ------------------------------------------------------------------------------
# 7. Live Vision Loop (Runs when session is active)
# ------------------------------------------------------------------------------
if st.session_state.is_running:
    cap = cv2.VideoCapture(0)
    last_frame_time = time.time()
    last_notification_time = 0
    is_away = False

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

        # --- A. Pose & Gaze Analysis ---
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

                # Handle Calibration Request from UI
                if getattr(st.session_state, "request_calibration", False):
                    st.session_state.base_nose_y = nose_y
                    st.session_state.base_shoulder_y = curr_shoulder_y
                    st.session_state.base_shoulder_width = curr_shoulder_width
                    st.session_state.is_calibrated = True
                    st.session_state.request_calibration = False
                    send_os_notification("Calibration Complete", "Posture baseline set successfully.")

                # Posture Evaluation
                if st.session_state.is_calibrated:
                    b_width = st.session_state.base_shoulder_width
                    nose_drop = (nose_y - st.session_state.base_nose_y) / b_width
                    shoulder_drop = (curr_shoulder_y - st.session_state.base_shoulder_y) / b_width
                    width_change = abs(curr_shoulder_width - b_width) / b_width

                    if nose_drop > POSTURE_SENSITIVITY or shoulder_drop > POSTURE_SENSITIVITY or width_change > (POSTURE_SENSITIVITY * 1.5):
                        is_bad_posture = True
                        if not st.session_state.is_on_break:
                            st.session_state.total_bad_posture += dt

                # Gaze Evaluation
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

        # --- B. Away From Desk Tracking ---
        if not person_detected and not st.session_state.is_on_break:
            st.session_state.total_away += dt

        # --- C. Phone Detection ---
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

        # --- D. OS Notifications ---
        if not st.session_state.is_on_break and (current_time - last_notification_time > NOTIFICATION_COOLDOWN):
            if is_bad_posture:
                send_os_notification("Posture Warning", "Bad posture detected! Please sit up straight.")
                last_notification_time = current_time
            elif phone_detected:
                send_os_notification("Distraction Warning", "Phone detected! Put your phone away.")
                last_notification_time = current_time

        # --- E. UI Rendering Updates ---
        if st.session_state.is_on_break:
            draw_text_clean(annotated_frame, "STATUS: ON BREAK ☕", (20, 40), font_scale=0.7, color=(255, 165, 0), thickness=2)

        # Convert BGR frame (OpenCV) to RGB (Streamlit requirement)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update Metrics Cards in Real-time
        metric_posture.metric("Bad Posture Time", f"{int(st.session_state.total_bad_posture)}s")
        metric_phone.metric("Phone Usage Time", f"{int(st.session_state.total_phone)}s")
        metric_gaze.metric("Looking Away Time", f"{int(st.session_state.total_looking_away)}s")
        metric_away.metric("Away From Desk Time", f"{int(st.session_state.total_away)}s")

    cap.release()