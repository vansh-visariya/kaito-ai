import pytest
from fastapi.testclient import TestClient
from api import app
from database.users import register, login, create_session

client = TestClient(app)

@pytest.fixture
def auth_cookies():
    # Register and login a test user
    try:
        register("api_test@example.com", "password123", "api_tester")
    except ValueError:
        pass # Already exists
    
    user = login("api_test@example.com", "password123")
    token = create_session(user["id"])
    return {"session_token": token}

def test_auth_status_unauthenticated():
    response = client.get("/api/auth/status")
    assert response.status_code == 401

def test_auth_status_authenticated(auth_cookies):
    response = client.get("/api/auth/status", cookies=auth_cookies)
    assert response.status_code == 200
    assert response.json()["authenticated"] is True
    assert response.json()["username"] == "api_tester"

def test_create_thread(auth_cookies):
    response = client.post("/api/threads/new", cookies=auth_cookies)
    assert response.status_code == 200
    data = response.json()
    assert "thread_id" in data
    
    # List threads should include this
    res_list = client.get("/api/threads", cookies=auth_cookies)
    threads = res_list.json()["threads"]
    assert any(t["id"] == data["thread_id"] for t in threads)

def test_branch_thread_invalid(auth_cookies):
    # Testing branching on a non-existent thread
    response = client.post(
        "/api/threads/branch", 
        json={"thread_id": "invalid_thread", "edit_index": 0},
        cookies=auth_cookies
    )
    assert response.status_code == 404
