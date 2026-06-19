"""User accounts with bcrypt-equivalent auth via pbkdf2_hmac."""

import hashlib
import os
import sqlite3
from pathlib import Path

DB_PATH = Path("database/users.db")


def _init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            salt BLOB NOT NULL,
            username TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tokens_used_today INTEGER DEFAULT 0,
            last_token_reset_date TEXT DEFAULT ''
        )
    """)
    # Migration for existing DB
    try:
        conn.execute("ALTER TABLE users ADD COLUMN tokens_used_today INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE users ADD COLUMN last_token_reset_date TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def _hash_password(password: str, salt: bytes = None):
    if salt is None:
        salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return key, salt


def register(email: str, password: str, username: str):
    """Register a new user. Returns True on success or raises ValueError."""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        if email and conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone():
            raise ValueError("Email already registered")
        pw_hash, salt = _hash_password(password)
        conn.execute(
            "INSERT INTO users (email, password_hash, salt, username) VALUES (?, ?, ?, ?)",
            (email, pw_hash, salt, username),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def login(email: str, password: str):
    """Validate credentials. Returns user dict or None."""
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT id, email, username, password_hash, salt FROM users WHERE email = ?",
        (email,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    uid, user_email, username, pw_hash, salt = row
    key, _ = _hash_password(password, salt)
    if key != pw_hash:
        return None
    return {"id": uid, "email": user_email, "username": username}


def create_session(user_id: int):
    """Create a new session token for the user."""
    import secrets
    token = secrets.token_hex(32)
    expires_ts = _future_timestamp(7 * 86400)  # 7 days
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
        (user_id, token, expires_ts),
    )
    conn.commit()
    conn.close()
    return token


def _future_timestamp(seconds: int) -> str:
    import datetime
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)).isoformat()


def validate_session(token: str):
    """Validate a session token. Returns user dict or None if expired/invalid."""
    import datetime
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        """SELECT s.user_id, u.email, u.username, s.expires_at
           FROM sessions s JOIN users u ON s.user_id = u.id
           WHERE s.token = ?""",
        (token,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    user_id, email, username, expires_at = row
    # Check expiry
    try:
        exp = datetime.datetime.fromisoformat(expires_at)
        if exp < datetime.datetime.now(datetime.timezone.utc):
            logout(token)
            return None
    except (ValueError, TypeError):
        pass
    return {"id": user_id, "email": email, "username": username}


def get_user_by_id(user_id: int):
    """Retrieve user info by ID. Returns user dict or None."""
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT id, email, username FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "email": row[1], "username": row[2]}


def logout(token: str):
    """Delete a session token."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def check_rate_limit(user_id: int, limit: int = 50000):
    """Check if user has exceeded their daily token limit."""
    import datetime
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT tokens_used_today, last_token_reset_date FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()
    
    if not row:
        conn.close()
        return

    tokens_used, last_reset = row
    today = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
    
    if last_reset != today:
        conn.execute(
            "UPDATE users SET tokens_used_today = 0, last_token_reset_date = ? WHERE id = ?",
            (today, user_id)
        )
        conn.commit()
        tokens_used = 0
        
    conn.close()
    
    if tokens_used >= limit:
        raise ValueError(f"Daily token limit ({limit}) exceeded. Please try again tomorrow.")


def increment_tokens(user_id: int, tokens: int):
    """Add tokens to the user's daily usage."""
    import datetime
    conn = sqlite3.connect(str(DB_PATH))
    today = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
    
    # Auto-reset if needed
    row = conn.execute("SELECT last_token_reset_date FROM users WHERE id = ?", (user_id,)).fetchone()
    if row and row[0] != today:
        conn.execute(
            "UPDATE users SET tokens_used_today = ?, last_token_reset_date = ? WHERE id = ?",
            (tokens, today, user_id)
        )
    else:
        conn.execute(
            "UPDATE users SET tokens_used_today = tokens_used_today + ? WHERE id = ?",
            (tokens, user_id)
        )
        
    conn.commit()
    conn.close()
