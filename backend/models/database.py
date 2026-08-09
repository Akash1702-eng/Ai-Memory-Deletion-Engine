"""
Minimal local SQLite database for model version tracking and training data management.
"""

import os
import sqlite3
import json
import uuid
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = "./data/local_app.db"


def init_local_db():
    """Initialize the SQLite database with required tables."""
    os.makedirs("./data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        id TEXT PRIMARY KEY,
        version TEXT UNIQUE,
        checkpoint_path TEXT,
        description TEXT,
        parent_version TEXT,
        is_active BOOLEAN DEFAULT 0,
        training_loss REAL,
        validation_loss REAL,
        metadata TEXT DEFAULT '{}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        category TEXT DEFAULT 'general',
        split TEXT DEFAULT 'train',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS unlearning_logs (
        id TEXT PRIMARY KEY,
        category TEXT,
        num_samples_forgotten INT,
        model_version_before TEXT,
        model_version_after TEXT,
        loss_before REAL,
        loss_after REAL,
        epochs_run INT,
        duration_seconds REAL,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS forgotten_records (
        record_id INTEGER NOT NULL,
        forget_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (record_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        trial_ends_at TIMESTAMP NOT NULL,
        subscription_status TEXT DEFAULT 'trial',
        subscription_plan TEXT DEFAULT 'free',
        subscription_expires_at TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS otp_codes (
        email TEXT PRIMARY KEY,
        otp TEXT NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Insert default compute_target = 'kaggle' if not exists
    cursor.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('compute_target', 'kaggle')")

    # Insert initial model version if not exists
    cursor.execute("SELECT COUNT(*) FROM model_versions WHERE version='v1.0'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
        INSERT INTO model_versions (id, version, checkpoint_path, description, is_active, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            "v1.0",
            "./models/base",
            "Base Qwen2.5-1.5B-Instruct model (pre-fine-tuning)",
            1,
            '{"model_name": "Qwen/Qwen2.5-1.5B-Instruct", "type": "base"}',
        ))

    conn.commit()
    conn.close()
    logger.info("Local database initialized at %s", DB_PATH)


def get_db_connection():
    """Return a SQLite connection with row factory."""
    init_local_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Model Versions ────────────────────────────────────────────────────────────

def save_model_version(
    version: str,
    checkpoint_path: str,
    description: str,
    parent_version: str,
    training_loss: float = None,
    validation_loss: float = None,
    metadata: dict = None,
):
    """Record a new model version in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("UPDATE model_versions SET is_active = 0 WHERE is_active = 1")

    cursor.execute("""
    INSERT OR REPLACE INTO model_versions
        (id, version, checkpoint_path, description, parent_version,
         is_active, training_loss, validation_loss, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        version,
        checkpoint_path,
        description,
        parent_version,
        1,
        training_loss,
        validation_loss,
        json.dumps(metadata or {}),
    ))

    conn.commit()
    conn.close()
    logger.info("Model version %s saved to database.", version)


def get_active_version() -> str:
    """Return the currently active model version string."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM model_versions WHERE is_active = 1")
    row = cursor.fetchone()
    conn.close()
    return row["version"] if row else "v1.0"


# ── Training Records ─────────────────────────────────────────────────────────

def save_training_records(records: list[dict]):
    """Save parsed CSV records to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Clear old records and stale forgotten markers
    cursor.execute("DELETE FROM training_records")
    cursor.execute("DELETE FROM forgotten_records")

    for rec in records:
        cursor.execute("""
        INSERT INTO training_records (question, answer, category, split)
        VALUES (?, ?, ?, ?)
        """, (
            rec.get("question", ""),
            rec.get("answer", ""),
            rec.get("category", "general"),
            rec.get("split", "train"),
        ))

    conn.commit()
    conn.close()
    logger.info("Saved %d training records to database.", len(records))


def get_training_records() -> list[dict]:
    """Return all training records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, question, answer, category, split FROM training_records ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_training_records_by_ids(ids: list[int]) -> list[dict]:
    """Return training records matching the given IDs."""
    if not ids:
        return []
    conn = get_db_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in ids)
    cursor.execute(
        f"SELECT id, question, answer, category, split FROM training_records WHERE id IN ({placeholders})",
        ids,
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_training_records_excluding_ids(ids: list[int]) -> list[dict]:
    """Return training records NOT in the given IDs."""
    if not ids:
        return get_training_records()
    conn = get_db_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in ids)
    cursor.execute(
        f"SELECT id, question, answer, category, split FROM training_records WHERE id NOT IN ({placeholders})",
        ids,
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ── Unlearning Logs ──────────────────────────────────────────────────────────

def save_unlearning_log(
    category: str,
    num_samples: int,
    version_before: str,
    version_after: str,
    loss_before: float,
    loss_after: float,
    epochs: int,
    duration: float,
):
    """Record an unlearning run in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO unlearning_logs
        (id, category, num_samples_forgotten, model_version_before,
         model_version_after, loss_before, loss_after, epochs_run,
         duration_seconds, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        category,
        num_samples,
        version_before,
        version_after,
        loss_before,
        loss_after,
        epochs,
        duration,
        "completed",
    ))

    conn.commit()
    conn.close()


# ── Forgotten Records ────────────────────────────────────────────────────────

def mark_records_as_forgotten(record_ids: list[int], forget_texts: Optional[list[str]] = None):
    """Mark training record IDs as forgotten so memory injection excludes them."""
    if not record_ids:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    for i, rid in enumerate(record_ids):
        text = forget_texts[i] if forget_texts and i < len(forget_texts) else None
        cursor.execute(
            "INSERT OR REPLACE INTO forgotten_records (record_id, forget_text) VALUES (?, ?)",
            (rid, text),
        )
    conn.commit()
    conn.close()
    logger.info("Marked %d records as forgotten.", len(record_ids))


def get_forgotten_record_ids() -> set[int]:
    """Return the set of record IDs that have been forgotten."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT record_id FROM forgotten_records")
    ids = {row["record_id"] for row in cursor.fetchall()}
    conn.close()
    return ids


def is_record_forgotten(record_id: int) -> bool:
    """Check if a specific record has been forgotten."""
    return record_id in get_forgotten_record_ids()


def clear_forgotten_records():
    """Clear all forgotten record markers (e.g. after retraining)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM forgotten_records")
    conn.commit()
    conn.close()
    logger.info("Cleared all forgotten record markers.")


def get_forgotten_texts() -> list[str]:
    """Return the forget_text strings from all forgotten records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT forget_text FROM forgotten_records WHERE forget_text IS NOT NULL")
    texts = [row["forget_text"] for row in cursor.fetchall()]
    conn.close()
    return texts


# ── App Settings (Compute Target) ────────────────────────────────────────────

def get_compute_target() -> str:
    """Return current compute target ('kaggle' or 'local')."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM app_settings WHERE key='compute_target'")
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else "kaggle"


def set_compute_target(target: str) -> str:
    """Set compute target ('kaggle' or 'local')."""
    target = target.lower().strip()
    if target not in ("kaggle", "local"):
        raise ValueError("Target must be 'kaggle' or 'local'")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES ('compute_target', ?, CURRENT_TIMESTAMP)",
        (target,),
    )
    conn.commit()
    conn.close()
    logger.info("Compute target updated to: %s", target)
    return target


# ── User Accounts & Subscriptions ───────────────────────────────────────────

def create_user_record(email: str, password_hash: str) -> dict:
    """Create a new user with 7 days of free trial."""
    from datetime import datetime, timedelta
    user_id = str(uuid.uuid4())
    now = datetime.utcnow()
    trial_ends = now + timedelta(days=7)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO users (id, email, password_hash, created_at, trial_ends_at, subscription_status, subscription_plan)
    VALUES (?, ?, ?, ?, ?, 'trial', 'free_trial_7_days')
    """, (user_id, email.lower().strip(), password_hash, now.isoformat(), trial_ends.isoformat()))
    conn.commit()
    conn.close()

    return {
        "id": user_id,
        "email": email.lower().strip(),
        "created_at": now.isoformat(),
        "trial_ends_at": trial_ends.isoformat(),
        "subscription_status": "trial",
        "subscription_plan": "free_trial_7_days",
    }


def get_user_by_email(email: str) -> Optional[dict]:
    """Retrieve user record by email."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=?", (email.lower().strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Retrieve user record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def check_user_subscription(email_or_id: str) -> dict:
    """Check subscription/trial status for user."""
    from datetime import datetime
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=? OR id=?", (email_or_id.lower().strip(), email_or_id))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"status": "unregistered", "active": False, "days_left": 0}

    user = dict(row)
    now = datetime.utcnow()

    # Check active paid subscription
    if user.get("subscription_status") == "active" and user.get("subscription_expires_at"):
        exp = datetime.fromisoformat(user["subscription_expires_at"])
        if now < exp:
            days_left = max(1, (exp - now).days)
            return {
                "status": "active",
                "active": True,
                "days_left": days_left,
                "plan": user.get("subscription_plan"),
                "expires_at": user["subscription_expires_at"],
            }

    # Check 7-day trial
    trial_ends = datetime.fromisoformat(user["trial_ends_at"])
    if now < trial_ends:
        days_left = max(1, (trial_ends - now).days)
        return {
            "status": "trial",
            "active": True,
            "days_left": days_left,
            "plan": "free_trial_7_days",
            "expires_at": user["trial_ends_at"],
        }

    return {
        "status": "expired",
        "active": False,
        "days_left": 0,
        "plan": user.get("subscription_plan"),
        "expires_at": user.get("subscription_expires_at"),
    }


def update_user_subscription(email_or_id: str, plan_id: str, months: int) -> dict:
    """Activate or extend user subscription."""
    from datetime import datetime, timedelta
    user = get_user_by_email(email_or_id) or get_user_by_id(email_or_id)
    if not user:
        raise ValueError("User not found")

    now = datetime.utcnow()
    expires_at = now + timedelta(days=30 * months)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE users
    SET subscription_status='active', subscription_plan=?, subscription_expires_at=?
    WHERE id=?
    """, (plan_id, expires_at.isoformat(), user["id"]))
    conn.commit()
    conn.close()

    return {
        "user_id": user["id"],
        "email": user["email"],
        "subscription_status": "active",
        "subscription_plan": plan_id,
        "expires_at": expires_at.isoformat(),
    }


def save_otp_code(email: str, otp: str, expires_minutes: int = 10):
    """Save an OTP code for an email with expiration time."""
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    expires_at = now + timedelta(minutes=expires_minutes)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO otp_codes (email, otp, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (email.lower().strip(), otp.strip(), expires_at.isoformat(), now.isoformat()),
    )
    conn.commit()
    conn.close()


def verify_otp_code(email: str, otp: str) -> bool:
    """Verify if the OTP matches and has not expired."""
    from datetime import datetime
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM otp_codes WHERE email=?", (email.lower().strip(),))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return False

    stored_otp = row["otp"]
    expires_at = datetime.fromisoformat(row["expires_at"])
    now = datetime.utcnow()

    if now > expires_at:
        cursor.execute("DELETE FROM otp_codes WHERE email=?", (email.lower().strip(),))
        conn.commit()
        conn.close()
        return False

    if stored_otp == otp.strip():
        # Clear used OTP
        cursor.execute("DELETE FROM otp_codes WHERE email=?", (email.lower().strip(),))
        conn.commit()
        conn.close()
        return True

    conn.close()
    return False
