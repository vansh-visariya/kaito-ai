import os
import pytest
from database.users import register, login, validate_session, create_session, DB_PATH
import sqlite3

@pytest.fixture(autouse=True)
def setup_db():
    # Use a test db for tests
    old_path = str(DB_PATH)
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(old_path)
    # clean up tables
    conn.execute("DELETE FROM sessions")
    conn.execute("DELETE FROM users")
    conn.commit()
    conn.close()
    yield
    # clean up after
    conn = sqlite3.connect(old_path)
    conn.execute("DELETE FROM sessions")
    conn.execute("DELETE FROM users")
    conn.commit()
    conn.close()

def test_register_and_login():
    # Test registration
    register("test@example.com", "password123", "testuser")
    
    # Test invalid login
    assert login("test@example.com", "wrongpassword") is None
    
    # Test valid login
    user = login("test@example.com", "password123")
    assert user is not None
    assert user["username"] == "testuser"
    assert user["email"] == "test@example.com"
    
def test_session_validation():
    register("test2@example.com", "password123", "testuser2")
    user = login("test2@example.com", "password123")
    
    token = create_session(user["id"])
    assert token is not None
    
    valid_user = validate_session(token)
    assert valid_user is not None
    assert valid_user["id"] == user["id"]
    
def test_rate_limiting():
    from database.users import increment_tokens, check_rate_limit
    register("test3@example.com", "password123", "testuser3")
    user = login("test3@example.com", "password123")
    
    # Increment tokens
    increment_tokens(user["id"], 49999)
    # Should not raise
    check_rate_limit(user["id"], 50000)
    
    increment_tokens(user["id"], 2)
    with pytest.raises(ValueError, match="Daily token limit"):
        check_rate_limit(user["id"], 50000)
