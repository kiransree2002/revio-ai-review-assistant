"""
Database layer for Revio AI multi-tenant review platform.
Uses SQLite with enforced foreign keys, business-level tenancy isolation,
and support for public review collection and employee authentication.
"""

import os
import sqlite3
from typing import Optional, List, Dict, Any
from src.auth import hash_password, verify_password

DEFAULT_DB_PATH = os.environ.get(
    "REVIO_DB_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "revio.db"
    )
)


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Establish a connection to the SQLite database with foreign keys enabled
    and dictionary-like row access.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize SQLite schema for multi-tenant businesses, employee users, and reviews.
    Safely migrates existing reviews tables with response approval workflow fields.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            # 1. Businesses table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS businesses (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Users table (employees associated with a specific business)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    business_id TEXT NOT NULL,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'employee',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE
                );
            """)

            # 3. Reviews table (stores customer submissions partitioned by business_id)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    business_id TEXT NOT NULL,
                    customer_name TEXT,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    review_text TEXT NOT NULL,
                    sentiment TEXT,
                    confidence REAL,
                    topic TEXT,
                    priority TEXT,
                    suggested_response TEXT,
                    response_status TEXT NOT NULL DEFAULT 'pending_approval' CHECK (response_status IN ('pending_approval', 'approved')),
                    approved_response TEXT,
                    approved_by INTEGER,
                    approved_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE,
                    FOREIGN KEY (approved_by) REFERENCES users(id) ON DELETE SET NULL
                );
            """)

            # Safe database migration: check and add missing columns for existing databases
            cursor = conn.execute("PRAGMA table_info(reviews);")
            columns = [row["name"] for row in cursor.fetchall()]

            if "response_status" not in columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN response_status TEXT NOT NULL DEFAULT 'pending_approval';")
            if "approved_response" not in columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN approved_response TEXT;")
            if "approved_by" not in columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN approved_by INTEGER;")
            if "approved_at" not in columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN approved_at TIMESTAMP;")

            # Indexes for efficient tenant queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_business ON users(business_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_business ON reviews(business_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_priority ON reviews(business_id, priority);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(business_id, response_status);")
    finally:
        conn.close()


def create_business(business_id: str, name: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Create a new business record.
    """
    if not business_id or not business_id.strip():
        raise ValueError("business_id cannot be empty.")
    if not name or not name.strip():
        raise ValueError("Business name cannot be empty.")
    
    clean_id = business_id.strip().lower()
    clean_name = name.strip()

    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO businesses (id, name) VALUES (?, ?)",
                (clean_id, clean_name)
            )
        return {"id": clean_id, "name": clean_name}
    except sqlite3.IntegrityError:
        raise ValueError(f"Business with ID '{clean_id}' already exists.")
    finally:
        conn.close()


def get_business(business_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Fetch a business record by ID.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute("SELECT * FROM businesses WHERE id = ?", (business_id.strip().lower(),))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_businesses(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Return all registered businesses.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute("SELECT * FROM businesses ORDER BY name ASC")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def create_user(
    business_id: str,
    username: str,
    password: str,
    role: str = "employee",
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Create an employee account linked to a valid business ID with salted PBKDF2 hash.
    """
    if not business_id or not business_id.strip():
        raise ValueError("business_id cannot be empty.")
    if not username or not username.strip():
        raise ValueError("username cannot be empty.")
    if not password or len(password) < 4:
        raise ValueError("password must be at least 4 characters long.")

    clean_business_id = business_id.strip().lower()
    clean_username = username.strip().lower()
    pwd_hash = hash_password(password)

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO users (business_id, username, password_hash, role)
                VALUES (?, ?, ?, ?)
                """,
                (clean_business_id, clean_username, pwd_hash, role)
            )
            user_id = cursor.lastrowid
        return {
            "id": user_id,
            "business_id": clean_business_id,
            "username": clean_username,
            "role": role
        }
    except sqlite3.IntegrityError as err:
        err_msg = str(err).lower()
        if "foreign key" in err_msg:
            raise ValueError(f"Cannot create user: Business '{clean_business_id}' does not exist.")
        elif "unique" in err_msg:
            raise ValueError(f"Username '{clean_username}' is already taken.")
        raise
    finally:
        conn.close()


def authenticate_user(username: str, password: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Authenticate employee credentials.
    Returns user dict with associated business details if verified, None otherwise.
    """
    if not username or not password:
        return None

    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT u.id, u.business_id, u.username, u.password_hash, u.role, b.name as business_name
            FROM users u
            JOIN businesses b ON u.business_id = b.id
            WHERE u.username = ?
            """,
            (username.strip().lower(),)
        )
        row = cursor.fetchone()
        if not row:
            return None

        if verify_password(password, row["password_hash"]):
            return {
                "id": row["id"],
                "business_id": row["business_id"],
                "business_name": row["business_name"],
                "username": row["username"],
                "role": row["role"]
            }
        return None
    finally:
        conn.close()


def save_review(
    business_id: str,
    rating: int,
    review_text: str,
    customer_name: Optional[str] = None,
    sentiment: Optional[str] = None,
    confidence: Optional[float] = None,
    topic: Optional[str] = None,
    priority: Optional[str] = None,
    suggested_response: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Persist a customer review along with optional pre-computed AI analysis results.
    Enforces business validation, input length limits, and rating bounds (1-5).
    """
    if not business_id or not business_id.strip():
        raise ValueError("business_id cannot be empty.")
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        raise ValueError("rating must be an integer between 1 and 5.")
    if not review_text or not review_text.strip():
        raise ValueError("review_text cannot be empty.")
    if len(review_text) > 5000:
        raise ValueError("review_text cannot exceed 5000 characters.")
    if customer_name and len(customer_name) > 100:
        raise ValueError("customer_name cannot exceed 100 characters.")

    clean_business_id = business_id.strip().lower()
    clean_customer_name = customer_name.strip() if customer_name and customer_name.strip() else None

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO reviews (
                    business_id, customer_name, rating, review_text,
                    sentiment, confidence, topic, priority, suggested_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_business_id,
                    clean_customer_name,
                    rating,
                    review_text.strip(),
                    sentiment,
                    confidence,
                    topic,
                    priority,
                    suggested_response
                )
            )
            review_id = cursor.lastrowid
        return {
            "id": review_id,
            "business_id": clean_business_id,
            "customer_name": clean_customer_name,
            "rating": rating,
            "review_text": review_text.strip(),
            "sentiment": sentiment,
            "confidence": confidence,
            "topic": topic,
            "priority": priority,
            "suggested_response": suggested_response,
            "response_status": "pending_approval",
            "approved_response": None,
            "approved_by": None,
            "approved_at": None
        }
    except sqlite3.IntegrityError as err:
        if "foreign key" in str(err).lower():
            raise ValueError(f"Cannot save review: Business '{clean_business_id}' does not exist.")
        raise
    finally:
        conn.close()


def get_reviews_by_business(business_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Retrieve all reviews strictly partitioned for a specific business ID.
    Enforces tenant privacy and returns reviews in descending chronological order.
    """
    if not business_id:
        return []

    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT r.*, u.username as approved_by_username 
            FROM reviews r
            LEFT JOIN users u ON r.approved_by = u.id
            WHERE r.business_id = ? 
            ORDER BY r.created_at DESC, r.id DESC
            """,
            (business_id.strip().lower(),)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def update_suggested_response(
    review_id: int,
    business_id: str,
    new_suggested_response: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Update the AI-suggested response (e.g. upon employee regenerating suggestion).
    Guards:
    - Business-level tenant isolation check
    - Does NOT overwrite an already approved response
    - Validates non-empty response
    """
    if not isinstance(review_id, int):
        raise ValueError("review_id must be an integer.")
    if not business_id or not business_id.strip():
        raise ValueError("business_id cannot be empty.")
    if not new_suggested_response or not new_suggested_response.strip():
        raise ValueError("Suggested response cannot be empty.")

    clean_business_id = business_id.strip().lower()
    clean_response = new_suggested_response.strip()

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                SELECT id, business_id, response_status, approved_response 
                FROM reviews 
                WHERE id = ? AND business_id = ?
                """,
                (review_id, clean_business_id)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError("Review not found or unauthorized business access.")

            if row["response_status"] == "approved":
                raise ValueError("Cannot overwrite suggested response for an already approved review.")

            conn.execute(
                """
                UPDATE reviews 
                SET suggested_response = ? 
                WHERE id = ? AND business_id = ?
                """,
                (clean_response, review_id, clean_business_id)
            )
        return {"id": review_id, "suggested_response": clean_response}
    finally:
        conn.close()


def approve_review_response(
    review_id: int,
    business_id: str,
    employee_user_id: int,
    approved_response: str,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Approve an AI-generated or employee-edited response.
    Guards:
    - Business-level tenant isolation check
    - Validates that approved_response is non-empty
    - Validates employee_user_id exists and belongs to the same business
    - Records approved_by employee ID, approved_at timestamp, and sets response_status = 'approved'
    """
    if not isinstance(review_id, int):
        raise ValueError("review_id must be an integer.")
    if not business_id or not business_id.strip():
        raise ValueError("business_id cannot be empty.")
    if not isinstance(employee_user_id, int):
        raise ValueError("employee_user_id must be an integer.")
    if not approved_response or not approved_response.strip():
        raise ValueError("Approved response cannot be empty.")

    clean_business_id = business_id.strip().lower()
    clean_approved_response = approved_response.strip()

    conn = get_db_connection(db_path)
    try:
        with conn:
            # Verify employee belongs to this business
            user_cursor = conn.execute(
                "SELECT id FROM users WHERE id = ? AND business_id = ?",
                (employee_user_id, clean_business_id)
            )
            if not user_cursor.fetchone():
                raise ValueError("Unauthorized employee or employee does not belong to this business.")

            # Verify review belongs to this business
            rev_cursor = conn.execute(
                "SELECT id FROM reviews WHERE id = ? AND business_id = ?",
                (review_id, clean_business_id)
            )
            if not rev_cursor.fetchone():
                raise ValueError("Review not found or unauthorized business access.")

            conn.execute(
                """
                UPDATE reviews 
                SET response_status = 'approved',
                    approved_response = ?,
                    approved_by = ?,
                    approved_at = CURRENT_TIMESTAMP
                WHERE id = ? AND business_id = ?
                """,
                (clean_approved_response, employee_user_id, review_id, clean_business_id)
            )

        return {
            "id": review_id,
            "response_status": "approved",
            "approved_response": clean_approved_response,
            "approved_by": employee_user_id
        }
    finally:
        conn.close()


def seed_demo_data(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Seed a safe demo business and demo employee account for local testing
    if they do not already exist. Uses a default test password that can be
    overridden via the REVIO_DEMO_PASSWORD environment variable.
    """
    init_db(db_path)
    demo_biz_id = "demo-corp"
    demo_biz_name = "Demo Retail Corp"
    demo_username = "demo_employee"
    demo_password = os.environ.get("REVIO_DEMO_PASSWORD", "demo1234")

    # Seed business if missing
    if not get_business(demo_biz_id, db_path):
        create_business(demo_biz_id, demo_biz_name, db_path)

    # Seed user if missing
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute("SELECT id FROM users WHERE username = ?", (demo_username,))
        if not cursor.fetchone():
            create_user(demo_biz_id, demo_username, demo_password, role="employee", db_path=db_path)
    finally:
        conn.close()

