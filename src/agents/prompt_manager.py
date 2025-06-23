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
                        new_msg = Message(to=INITIATOR_JID)
                        new_msg.set_metadata("phase", "final")
                        new_msg.body = json.dumps({"final_answer": final_answer})
                        await self.send(new_msg)
                        print("DistributedPromptAgent: Respuesta enviada")
                except Exception as e:
                    print(f"DistributedPromptAgent: Error - {str(e)}")
            else:
                await asyncio.sleep(1)

        async def generate_final_answer(self, query, context,params, max_new_tokens=250, num_beams=5, max_retries=3):
    
            prompt_template0 = """ 
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
                
            prompt_template1 = """[INST] Eres un asistente experto en historia que responde preguntas de manera altamente personalizada. 
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
                1. Respuesta basada ÚNICAMENTE en: {context}
                2. Adapta tono usando humor={humor_level} y formalidad={formality_level}
                3. Formato principal: {response_types}
                4. Máximo énfasis en: {preferred_topics} (alta afinidad)
                5. Evita ABSOLUTAMENTE: {disliked_topics}
                6. Aplica perspectiva: {historiographical_approach}
                7. Maneja controversias con: {controversy_handling}
                8. Usa enfoque temporal: {temporal_focus}
                9. Limita a {max_length} tokens

                ### Parámetros técnicos:
                - Temperature: {temperature} | Top_p: {top_p} | Repetition_penalty: {repetition_penalty}

                ### Pregunta:
                {query} 
                [/INST]

                Respuesta personalizada:"""

            prompt_template2 = """[INST] Eres un asistente experto en historia que responde preguntas de manera altamente personalizada. 
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
                1. Construye una narrativa usando: {context}
                2. Tonos combinados: humor={humor_level} + formalidad={formality_level}
                3. Estructura: {response_types}  
                4. Protagonistas: {preferred_topics} 
                5. Excluye: {disliked_topics}
                6. Lente historiográfico: {historiographical_approach}
                7. Controversias: {controversy_handling}
                8. Marco temporal: {temporal_focus}
                9. Máximo {max_length} tokens

                ### Parámetros técnicos:
                - Temperature: {temperature} | Top_p: {top_p} | Repetition_penalty: {repetition_penalty}

                ### Pregunta:
                {query} 
                [/INST]

                Respuesta personalizada:"""
            prompt_template3 = """[INST] Eres un asistente experto en historia que responde preguntas de manera altamente personalizada. 
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
                1. Analiza rigurosamente: {context}
                2. Ajuste comunicacional: humor={humor_level} + formalidad={formality_level}
                3. Organiza en formato: {response_types}  
                4. Foco temático: {preferred_topics} 
                5. Omite: {disliked_topics}
                6. Metodología: {historiographical_approach}
                7. Protocolo controversias: {controversy_handling}
                8. Perspectiva temporal: {temporal_focus}
                9. {max_length} tokens máximo

                ### Parámetros técnicos:
                - Temperature: {temperature} | Top_p: {top_p} | Repetition_penalty: {repetition_penalty}

                ### Pregunta:
                {query} 
                [/INST]

                Respuesta personalizada:"""
            prompt_template4 = """[INST] Eres un asistente experto en historia que responde preguntas de manera altamente personalizada. 
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
                1. Trabaja en equipo usando: {context}
                2. Conexión emocional: humor={humor_level} + formalidad={formality_level}
                3. Diálogo en formato: {response_types}  
                4. Temas centrales: {preferred_topics} 
                5. Sensibilidad a: {disliked_topics} (evitar)
                6. Enfoque académico: {historiographical_approach}
                7. Controversias: {controversy_handling}
                8. Énfasis temporal: {temporal_focus}
                9. Respuesta concisa ({max_length} tokens)

                ### Parámetros técnicos:
                - Temperature: {temperature} | Top_p: {top_p} | Repetition_penalty: {repetition_penalty}

                ### Pregunta:
                {query} 
                [/INST]

                Respuesta personalizada:"""
            prompt_template5 = """[INST] Eres un asistente experto en historia que responde preguntas de manera altamente personalizada. 
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
                1. Respuesta PRECISA basada en: {context}
                2. Sincronización estilística: humor={humor_level} + formalidad={formality_level}
                3. Estructura: {response_types}  
                4. Priorizar: {preferred_topics} 
                5. Cero menciones a: {disliked_topics}
                6. Rigor metodológico: {historiographical_approach}
                7. Controversias: {controversy_handling}
                8. Contexto temporal: {temporal_focus}
                9. Extensión máxima: {max_length} tokens

                ### Parámetros técnicos:
                - Temperature: {temperature} | Top_p: {top_p} | Repetition_penalty: {repetition_penalty}

                ### Pregunta:
                {query} 
                [/INST]

                Respuesta personalizada:"""
            
            prompt_template = prompt_template1
            
            
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