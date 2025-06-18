import asyncio
import json
import os
import requests
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from utils.constants import INITIATOR_JID, HF_MODEL,MOODLE_JID

class PromptAgent(Agent):
    async def setup(self):
        self.hf_api_token = os.getenv("HF_API_TOKEN")
        self.add_behaviour(self.PromptBehaviour())
        self.params={}
        print(f"{self.jid} iniciado correctamente")

    class PromptBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(20)
            if msg and msg.get_metadata("phase")== "params":
                self.agent.params= json.loads(msg.body)
            elif msg and msg.get_metadata("phase") == "prompt":
                try:
                    data = json.loads(msg.body)
                    query = data.get("query", "")
                    context = data.get("context", "")
                    
                    print(f"DistributedPromptAgent: Generando respuesta para: {query}")
                    if self.agent.params:
                        final_answer = await self.generate_final_answer(query, context,self.agent.params)                    
                        new_msg = Message(to=MOODLE_JID)
                        new_msg.set_metadata("phase", "final")
                        new_msg.body = json.dumps({"final_answer": final_answer})
                        await self.send(new_msg)
                        print("DistributedPromptAgent: Respuesta enviada")
                except Exception as e:
                    print(f"DistributedPromptAgent: Error - {str(e)}")
            else:
                await asyncio.sleep(1)

        async def generate_final_answer(self, query, context,params, max_new_tokens=250, num_beams=5, max_retries=3):
    
            prompt_template = """ 
                [INST] Eres un asistente experto en historia que responde preguntas de manera altamente personalizada. 
                Sigue estrictamente estas directrices:
                ### Perfil de usuario y preferencias:
                - **Estilo de comunicación**: {style}
                - **Nivel de humor**: {humor_level} (0: serio, 1: humorístico)
                - **Formalidad**: {formality_level} (0: coloquial, 1: académico)
                - **Temas preferidos**: {preferred_topics}
                - **Temas evitados**: {disliked_topics}
                - **Tipos de respuesta preferidos**: {response_types}

                ### Enfoque histórico requerido:
                - **Perspectiva historiográfica**: {historiographical_approach}
                - **Tratamiento de fuentes**: {source_criticism}
                - **Preferencia de evidencia**: {evidence_preference}
                - **Manejo de controversias**: {controversy_handling}
                - **Enfoque temporal**: {temporal_focus}

                ### Instrucciones clave:
                1. Respuesta basada ÚNICAMENTE en el contexto proporcionado
                2. Adapta el tono según humor_level y formality_level
                3. Usa {response_types} como formato principal
                4. Enfatiza temas con alta affinity ({topic_affinity})
                5. Evita mencionar {disliked_topics}
                6. Aplica {historiographical_approach} al analizar eventos
                7. Maneja controversias con {controversy_handling}
                8. Limita respuesta a {max_length} tokens

                ### Contexto (base factual):
                {context}

                ### Pregunta del usuario:
                {query}

                ### Parámetros técnicos de generación:
                - Creatividad controlada (temperature: {temperature})
                - Enfoque léxico (top_p: {top_p})
                - Prevención de repetición (repetition_penalty: {repetition_penalty})
                [/INST]

                Respuesta personalizada:"""
            
            prompt = prompt_template.format(
                style=params['style'],
                humor_level=params['humor_level'],
                formality_level=params['formality_level'],
                preferred_topics=", ".join(params['preferred_topics']),
                disliked_topics=", ".join(params['disliked_topics']),
                response_types=", ".join(params['response_types']),
                historiographical_approach=params['historiographical_approach'],
                source_criticism=params['source_criticism'],
                evidence_preference=params['evidence_preference'],
                controversy_handling=params['controversy_handling'],
                temporal_focus=params['temporal_focus'],
                topic_affinity=params['topic_affinity'],
                max_length=params['max_length'],
                temperature=params['temperature'],
                top_p=params['top_p'],
                repetition_penalty=params['repetition_penalty'],
                context=context,
                query=query
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
    response.raise_for_status()
    return response.json()

def parse_response(response_data):
    if "message" in response_data and isinstance(response_data["message"], dict) and "content" in response_data["message"]:
        return response_data["message"]["content"]
    if ("choices" in response_data and isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0):
        first_choice = response_data["choices"][0]
        if ("message" in first_choice and isinstance(first_choice["message"], dict) and "content" in first_choice["message"]):
            return first_choice["message"]["content"]
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