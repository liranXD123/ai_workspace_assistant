import json
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# 1. טעינת מודל Ollama המקומי (למשל llama3)
print("🤖 מפעיל את מודל Ollama (Llama3)...")
llm = Ollama(model="llama3")


def load_session_data():
    """טוענת את נתוני המעקב האחרונים מקובץ ה-JSON"""
    if not os.path.exists('session_log.json'):
        return None

    try:
        with open('session_log.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"שגיאה בקריאת הקובץ: {e}")
        return None


def generate_analytics_report(question: str):
    """מייצרת תשובה חכמה מה-LLM בהתבסס על הנתונים הוויזואליים"""
    data = load_session_data()

    if not data:
        return "עדיין אין נתונים זמינים. בצע כיול והרץ את מנוע הראייה הממוחשבת (vision_engine.py)."

    template = """
    You are an expert AI Productivity & Ergonomics Assistant. 
    Below is the real-time analytics data collected from the user's webcam session:

    - Session Timestamp: {timestamp}
    - Total Bad Posture Time: {bad_posture_seconds} seconds
    - Total Phone Usage Time: {phone_seconds} seconds
    - Total Time Looking Away From Screen: {looking_away_seconds} seconds
    - Total Time Away From Desk: {away_from_desk_seconds} seconds

    Current Live Status:
    - Posture Status: {posture_status}
    - Phone Status: {phone_status}
    - Gaze Status: {gaze_status}

    User Question: {question}

    Provide a helpful, concise, and actionable response in English based on these exact metrics. 
    If the user asks for a summary or score, evaluate their focus and posture constructively.
    """

    prompt = PromptTemplate(
        input_variables=[
            "timestamp", "bad_posture_seconds", "phone_seconds",
            "looking_away_seconds", "away_from_desk_seconds",
            "posture_status", "phone_status", "gaze_status", "question"
        ],
        template=template
    )

    formatted_prompt = prompt.format(
        timestamp=data['timestamp'],
        bad_posture_seconds=data['metrics']['bad_posture_seconds'],
        phone_seconds=data['metrics']['phone_seconds'],
        looking_away_seconds=data['metrics']['looking_away_seconds'],
        away_from_desk_seconds=data['metrics']['away_from_desk_seconds'],
        posture_status=data['current_status']['posture'],
        phone_status=data['current_status']['phone'],
        gaze_status=data['current_status']['gaze'],
        question=question
    )

    response = llm.invoke(formatted_prompt)
    return response


# ------------------------------------------------------------------------------
# 💬 צ'אט אינטראקטיבי מול הסוכן
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("💬 AI Workspace Productivity Agent Ready!")
    print("שאל את הסוכן שאלות (למשל: 'How was my posture today?', 'Give me a productivity score').")
    print("הקלד 'exit' כדי לצאת.")
    print("=" * 60 + "\n")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        print("\n🤔 Thinking...")
        answer = generate_analytics_report(user_query)
        print(f"\nAI Agent:\n{answer}")