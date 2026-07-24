import cv2
import numpy as np
import time
import json
import os
from ultralytics import YOLO
from plyer import notification
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ------------------------------------------------------------------------------
# 1. טעינת מודלים
# ------------------------------------------------------------------------------
pose_model = YOLO('yolov8n-pose.pt')
detect_model = YOLO('yolov8n.pt')

POSTURE_SENSITIVITY = 0.12
NOTIFICATION_COOLDOWN = 10


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
    """טקסט נקי בלבד ללא צלליות או מלבנים"""
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def generate_end_of_session_report(session_data):
    print("\n" + "=" * 60)
    print("⏳ Generating End-of-Session AI Summary Report via Ollama...")
    print("=" * 60)

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
        # שימוש במחלקה המעודכנת OllamaLLM
        llm = OllamaLLM(model="llama3")
        formatted_prompt = prompt.format(
            bad_posture_seconds=session_data['metrics']['bad_posture_seconds'],
            phone_seconds=session_data['metrics']['phone_seconds'],
            looking_away_seconds=session_data['metrics']['looking_away_seconds'],
            away_from_desk_seconds=session_data['metrics']['away_from_desk_seconds']
        )
        report = llm.invoke(formatted_prompt)

        print("\n📊 === AI SESSION SUMMARY REPORT ===")
        print(report)
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"❌ Error generating AI report: {e}")


def reset_session_log():
    """מאפסת את קובץ ה-JSON לקראת הסשן הבא"""
    empty_data = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "metrics": {
            "bad_posture_seconds": 0,
            "phone_seconds": 0,
            "looking_away_seconds": 0,
            "away_from_desk_seconds": 0
        },
        "current_status": {
            "posture": "Reset",
            "phone": "Reset",
            "gaze": "Reset"
        }
    }
    with open('session_log.json', 'w') as f:
        json.dump(empty_data, f, indent=4)
    print("🧹 Session log has been safely reset for the next run.")


# ------------------------------------------------------------------------------
# 2. אתחול משתנים וטיימרים
# ------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("שגיאה: לא ניתן לפתוח את המצלמה")
    exit()

is_calibrated = False
base_nose_y = 0
base_shoulder_y = 0
base_shoulder_width = 0

total_bad_posture_time = 0
total_phone_time = 0
total_looking_away_time = 0
total_away_time = 0

last_frame_time = time.time()
last_event_time = time.time()
last_notification_time = 0

is_away = False

print("=" * 60)
print("🚀 AI Workspace Assistant Engine is Running!")
print("1. שב ישר ולחץ 'c' במקלדת לכיול.")
print("2. בסיום הסשן לחץ 'q' לקבלת דוח AI אוטומטי ואיפוס הנתונים.")
print("=" * 60)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    dt = current_time - last_frame_time
    last_frame_time = current_time

    posture_status = "Not Calibrated (Press 'c')" if not is_calibrated else "Good"
    posture_color = (0, 215, 255) if not is_calibrated else (0, 255, 127)

    person_detected = False
    phone_detected = False
    is_bad_posture = False
    is_looking_away = False

    # ----------------------------------------------
    #  A. זיהוי שלד, יציבה ומבט
    # ----------------------------------------------
    pose_results = pose_model(frame, verbose=False)[0]
    annotated_frame = pose_results.plot()

    if pose_results.keypoints is not None and len(pose_results.keypoints.data) > 0:
        kpts = pose_results.keypoints.data[0].cpu().numpy()

        # אינדקסים ב-COCO Pose: 0=Nose, 1=L_Eye, 2=R_Eye, 3=L_Ear, 4=R_Ear, 5=L_Shoulder, 6=R_Shoulder
        if len(kpts) >= 7 and kpts[5][2] > 0.4 and kpts[6][2] > 0.4:
            person_detected = True

            nose_x, nose_y = kpts[0][:2]
            l_eye, r_eye = kpts[1][:2], kpts[2][:2]
            l_sh, r_sh = kpts[5][:2], kpts[6][:2]

            curr_shoulder_y = (l_sh[1] + r_sh[1]) / 2
            curr_shoulder_width = np.linalg.norm(l_sh - r_sh)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                base_nose_y = nose_y
                base_shoulder_y = curr_shoulder_y
                base_shoulder_width = curr_shoulder_width
                is_calibrated = True
                print("[CALIBRATION] יציבה כוילה בהצלחה!")
                send_os_notification("Calibration Complete", "Your posture baseline has been set successfully.")

            # --- בדיקת יציבה ---
            if is_calibrated:
                nose_drop = (nose_y - base_nose_y) / base_shoulder_width
                shoulder_drop = (curr_shoulder_y - base_shoulder_y) / base_shoulder_width
                width_change = abs(curr_shoulder_width - base_shoulder_width) / base_shoulder_width

                if nose_drop > POSTURE_SENSITIVITY or shoulder_drop > POSTURE_SENSITIVITY or width_change > (
                        POSTURE_SENSITIVITY * 1.5):
                    is_bad_posture = True
                    posture_status = "BAD POSTURE"
                    posture_color = (50, 50, 255)
                    total_bad_posture_time += dt

            # --- בדיקת מבט משודרגת (3D Look-Away: למעלה/למטה + צדדים) ---
            if kpts[0][2] > 0.4 and kpts[1][2] > 0.4 and kpts[2][2] > 0.4:
                eyes_center_x = (l_eye[0] + r_eye[0]) / 2
                eyes_center_y = (l_eye[1] + r_eye[1]) / 2
                eye_distance = abs(r_eye[0] - l_eye[0])

                if eye_distance > 0:
                    # 1. חישוב מבט אנכי (למעלה/למטה - Vertical Pitch)
                    eye_nose_dist_y = nose_y - eyes_center_y
                    eye_nose_ratio_y = eye_nose_dist_y / curr_shoulder_width
                    is_vertical_away = eye_nose_ratio_y < 0.03 or eye_nose_ratio_y > 0.22

                    # 2. חישוב מבט אופקי (צדדים - Horizontal Yaw)
                    # יחס המרחק האופקי של האף ממרכז העיניים ביחס למרחק בין העיניים
                    nose_offset_x = abs(nose_x - eyes_center_x)
                    horizontal_ratio = nose_offset_x / eye_distance

                    # אם האף זז הצידה יותר מ-35% ממרכז העיניים, או שאחת העיניים נעלמת/מוסתרת בגלל סיבוב ראש
                    is_horizontal_away = horizontal_ratio > 0.35

                    if is_vertical_away or is_horizontal_away:
                        is_looking_away = True
                        total_looking_away_time += dt

    # ----------------------------------------------
    #  B. מעקב קימה מהכיסא
    # ----------------------------------------------
    if not person_detected:
        if not is_away:
            is_away = True
        else:
            total_away_time += dt
    else:
        if is_away:
            is_away = False

    # ----------------------------------------------
    #  C. זיהוי טלפון
    # ----------------------------------------------
    detect_results = detect_model(frame, verbose=False)[0]
    for box in detect_results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 67 and conf > 0.4:
            phone_detected = True
            total_phone_time += dt
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (50, 50, 255), 2)

    # ----------------------------------------------
    #  D. התראות קופצות בלייב
    # ----------------------------------------------
    if current_time - last_notification_time > NOTIFICATION_COOLDOWN:
        if is_bad_posture:
            send_os_notification("Posture Warning", "Bad posture detected! Please sit up straight.")
            last_notification_time = current_time
        elif phone_detected:
            send_os_notification("Distraction Warning", "Phone detected! Put your phone away to stay focused.")
            last_notification_time = current_time

    # ----------------------------------------------
    #  E. עדכון קובץ JSON רציף
    # ----------------------------------------------
    if is_calibrated and current_time - last_event_time > 2.0:
        session_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": {
                "bad_posture_seconds": int(total_bad_posture_time),
                "phone_seconds": int(total_phone_time),
                "looking_away_seconds": int(total_looking_away_time),
                "away_from_desk_seconds": int(total_away_time)
            },
            "current_status": {
                "posture": posture_status,
                "phone": "PHONE IN USE" if phone_detected else "No Phone",
                "gaze": "LOOKING AWAY" if is_looking_away else "Focused"
            }
        }
        with open('session_log.json', 'w') as f:
            json.dump(session_data, f, indent=4)
        last_event_time = current_time

    # ----------------------------------------------
    #  F. הצגת טקסטים על המסך (נקי ללא צלליות)
    # ----------------------------------------------
    draw_text_clean(annotated_frame, f"Posture: {posture_status}", (20, 30), font_scale=0.65, color=posture_color,
                    thickness=2)

    phone_str = "PHONE IN USE" if phone_detected else "No Phone"
    draw_text_clean(annotated_frame, f"Phone: {phone_str}", (20, 55), font_scale=0.6,
                    color=(50, 50, 255) if phone_detected else (0, 255, 127))

    gaze_str = "LOOKING AWAY" if is_looking_away else "Focused"
    draw_text_clean(annotated_frame, f"Gaze: {gaze_str}", (20, 80), font_scale=0.6,
                    color=(50, 50, 255) if is_looking_away else (0, 255, 127))

    right_x = annotated_frame.shape[1] - 240
    draw_text_clean(annotated_frame, "--- SESSION ANALYTICS ---", (right_x, 30), font_scale=0.5, color=(200, 200, 200))
    draw_text_clean(annotated_frame, f"Bad Posture: {int(total_bad_posture_time)}s", (right_x, 50), font_scale=0.55,
                    color=(0, 215, 255))
    draw_text_clean(annotated_frame, f"Phone Time:  {int(total_phone_time)}s", (right_x, 70), font_scale=0.55,
                    color=(0, 215, 255))
    draw_text_clean(annotated_frame, f"Look Away:   {int(total_looking_away_time)}s", (right_x, 90), font_scale=0.55,
                    color=(0, 215, 255))
    draw_text_clean(annotated_frame, f"Away Desk:   {int(total_away_time)}s", (right_x, 110), font_scale=0.55,
                    color=(0, 215, 255))

    cv2.imshow('AI Workspace - Full Analytics Engine', annotated_frame)

    # ----------------------------------------------
    #  G. יציאה, הפקת דוח אוטומטי ואיפוס (מקש 'q')
    # ----------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        final_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": {
                "bad_posture_seconds": int(total_bad_posture_time),
                "phone_seconds": int(total_phone_time),
                "looking_away_seconds": int(total_looking_away_time),
                "away_from_desk_seconds": int(total_away_time)
            }
        }

        generate_end_of_session_report(final_data)
        reset_session_log()
        break

cap.release()
cv2.destroyAllWindows()