import requests
import json

API_URL = "https://apigateway.avangenio.net/chat/completions"
MODEL = "free"
API_KEY = "sk-0iV3NeVrYrdTYJsyEeygFZ3DyatEv6F9q6v8GDdC52KDADbZ"

def call_endpoint(body):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(API_URL, headers=headers, json=body)
    response.raise_for_status()  # Lanza una excepción si hay algún error
    return response.json()

def parse_response(response_data):
    # Se busca el campo "message" en la raíz.
    if "message" in response_data and isinstance(response_data["message"], dict) and "content" in response_data["message"]:
        return response_data["message"]["content"]
    # Sino, se revisa dentro de "choices"
    if ("choices" in response_data and isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0):
        first_choice = response_data["choices"][0]
        if ("message" in first_choice and isinstance(first_choice["message"], dict) and "content" in first_choice["message"]):
            return first_choice["message"]["content"]
    # Si no se encuentra la estructura deseada, retornar el JSON completo como cadena
    return json.dumps(response_data)

def send_request(prompt):
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }
    return call_endpoint(body)

if __name__ == "__main__":
    prompt = (
        "[INST] Eres un asistente experto que responde preguntas basado únicamente en el contexto proporcionado. "
        "Responde de manera concisa y precisa sin añadir información externa.\n\n"
        "Contexto: Influencia de la Revolución Francesa en las Independencias de América Latina - PanoramaCultural.com.co Historia\n"
        "Influencia de la Revolución Francesa en las Independencias de América Latina\n"
        "José Luis Hernández 14/07/2023 - 06:12\n"
        "¿Cuál es la influencia real que pudo haber tenido la Revolución Francesa en el Proceso de Independencia de América Latina, a pesar de la distancia geográfica que los separa?"
    )
    
    try:
        response_data = send_request(prompt)
        result = parse_response(response_data)
        print(result)  # Solo se imprime el contenido del mensaje
    except requests.exceptions.RequestException as err:
        print("Error realizando la solicitud:", err)
