import uuid
import requests

def generate_unique_id():
    return str(uuid.uuid4())

def validate_groq_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    try:
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
        return response.status_code == 200
    except Exception:
        return False

