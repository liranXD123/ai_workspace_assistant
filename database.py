import sqlite3 as sql

DB_NAME = "database.db"

def get_connection():
    """Establishes a connection to SQLite database and enables Foreign Key constraints."""
    conn = sql.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initializes the database schema for users and sessions if tables do not exist."""
    conn = get_connection()
    cur = conn.cursor()

    # Create Users table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            PASSWORD_HASH TEXT NOT NULL,
            CREATED_AT TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create Sessions table with Foreign Key linking to Users
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(ID) ON DELETE CASCADE,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            bad_posture_seconds INTEGER NOT NULL DEFAULT 0,
            phone_seconds INTEGER NOT NULL DEFAULT 0,
            looking_away INTEGER NOT NULL DEFAULT 0,
            away_seconds INTEGER NOT NULL DEFAULT 0,
            ai_report TEXT NOT NULL DEFAULT ''
        )
    ''')

    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# User Authentication Helpers
# ------------------------------------------------------------------------------

def create_user(username: str, password_hash: str) -> bool:
    """
    Registers a new user with a hashed password using parameterized query.
    Returns True if registration succeeds, False if username already exists.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, PASSWORD_HASH) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        return True
    except sql.IntegrityError:
        # Fails if username violates the UNIQUE constraint
        return False
    finally:
        conn.close()


def get_user_by_username(username: str):
    """
    Retrieves user record by username safely against SQL Injection.
    Returns tuple (id, username, password_hash, created_at) or None.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        return user
    finally:
        conn.close()


# ------------------------------------------------------------------------------
# Session Analytics Helpers
# ------------------------------------------------------------------------------

def save_session(user_id: int, bad_posture: int, phone: int, looking_away: int, away: int, ai_report: str) -> bool:
    """
    Saves a completed work session and its AI report linked to a specific user_id.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute('''
            INSERT INTO sessions (user_id, bad_posture_seconds, phone_seconds, looking_away, away_seconds, ai_report)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, bad_posture, phone, looking_away, away, ai_report))
        conn.commit()
        return True
    except sql.Error as e:
        print(f"[Database Error] Failed to save session: {e}")
        return False
    finally:
        conn.close()


def get_user_sessions(user_id: int):
    """
    Fetches all historical sessions for a user, ordered from newest to oldest.
    Returns list of tuples or an empty list.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute('''
            SELECT ID, timestamp, bad_posture_seconds, phone_seconds, looking_away, away_seconds, ai_report
            FROM sessions
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,))
        sessions = cur.fetchall()
        return sessions
    finally:
        conn.close()


# Initialize database when script is run directly
if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")