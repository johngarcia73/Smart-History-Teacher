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
            
            url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
            headers = {"Authorization": f"Bearer {self.agent.hf_api_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "return_full_text": False,
                    "max_new_tokens": max_new_tokens,
                    "num_beams": num_beams,
                    "early_stopping": True
                }
            }
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        return result[0].get("generated_text", str(result)).strip()
                    elif response.status_code == 503:
                        await asyncio.sleep(15 * (attempt + 1))
                    else:
                        print(f"Error en generación: {response.status_code} - {response.text}")
                        await asyncio.sleep(5)
                except Exception as e:
                    print(f"Error en conexión: {str(e)}")
                    await asyncio.sleep(5)
            
            return "No se pudo generar una respuesta. Por favor intenta nuevamente."