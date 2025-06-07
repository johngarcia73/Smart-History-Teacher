import requests
import os

def test_hf_api():
    prompt = "[INST] Responde brevemente: ¿Cuál es la capital de Francia? [/INST]\nRespuesta:"
    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
        headers={"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"},
        json={"inputs": prompt, "parameters": {"return_full_text": False}}
    )
    print("API Response:", response.status_code, response.json())

test_hf_api()