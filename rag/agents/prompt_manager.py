import asyncio
import json
import os
import requests
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from utils.constants import INITIATOR_JID, HF_MODEL

class PromptAgent(Agent):
    async def setup(self):
        self.hf_api_token = os.getenv("HF_API_TOKEN")
        self.add_behaviour(self.PromptBehaviour())
        print(f"{self.jid} iniciado correctamente")

    class PromptBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg and msg.get_metadata("phase") == "prompt":
                try:
                    data = json.loads(msg.body)
                    query = data.get("query", "")
                    context = data.get("context", "")
                    
                    print(f"DistributedPromptAgent: Generando respuesta para: {query}")
                    final_answer = await self.generate_final_answer(query, context)
                    
                    new_msg = Message(to=INITIATOR_JID)
                    new_msg.set_metadata("phase", "final")
                    new_msg.body = json.dumps({"final_answer": final_answer})
                    await self.send(new_msg)
                    print("DistributedPromptAgent: Respuesta enviada")
                except Exception as e:
                    print(f"DistributedPromptAgent: Error - {str(e)}")
            else:
                await asyncio.sleep(1)

        async def generate_final_answer(self, query, context, max_new_tokens=250, num_beams=5, max_retries=3):
            prompt = (
                "[INST] Eres un asistente experto que responde preguntas basado únicamente en el contexto proporcionado. "
                "Responde de manera concisa y precisa sin añadir información externa.\n\n"
                f"Contexto: {context}\n\n"
                f"Pregunta: {query} [/INST]\n"
                "Respuesta:"
            )
            
            
            for attempt in range(max_retries):
                try:
                    response = send_request(prompt)
                    result = parse_response(response)
                    return result
                except Exception as e:
                    print(f"Error en conexión: {str(e)}")
                    await asyncio.sleep(5)
            
            return "No se pudo generar una respuesta. Por favor intenta nuevamente."
        
def call_endpoint(body):
    headers = {
        "Authorization": f"Bearer sk-0iV3NeVrYrdTYJsyEeygFZ3DyatEv6F9q6v8GDdC52KDADbZ",
        "Content-Type": "application/json"
    }
    response = requests.post("https://apigateway.avangenio.net/chat/completions", headers=headers, json=body)
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
        "model": "free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }
    return call_endpoint(body)