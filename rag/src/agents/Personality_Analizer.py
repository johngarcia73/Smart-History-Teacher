from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

import json
import requests

class PersonalityAnalyzerAgent(Agent):
             

    

    class SearchProfileBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg and msg.metadata["phase"] == "analyzer":
                # Parsear mensaje entrante
                data = json.loads(msg.body)
                user_id = data["user_id"]
                raw_query = data["query"]
                
                # Obtener perfil actual del usuario
                msg_profile= Message(
                    to="profilemanageragent@localhost",
                    body=json.dumps({
                        "user_id": user_id,
                        "raw_query":raw_query
                        }),
                    metadata={"phase":"profile"}
                )
                await self.send(msg_profile)
    class AnalyzerBahaviour(CyclicBehaviour):
        async def run(self):
            msg_profile= await self.receive(timeout=30)
            if msg_profile and msg_profile.metadata["phase"] == "profile":
                data= json.loads(msg_profile.body)
                profile= data["profile"]
                # Construir prompt para análisis de personalidad y query
                prompt = await self._build_analysis_prompt(data["raw_query"], profile)

                # Obtener respuesta estructurada del LLM
                llm_response =await self.send_request(prompt)
   
                # Parsear respuesta estructurada
                structured_data = await self._parse_llm_response(llm_response,data["raw_query"])
   
                # Preparar mensaje para el siguiente agente
                next_msg = Message(to="search_agent@localhost")
                next_msg.set_metadata("phase", "query")
                next_msg.body = json.dumps({
                    "query": structured_data["expanded_query"]
                })
                await self.send(next_msg)
                next_msg = Message(
                    to= "profilemanageragent@localhost",
                    body=json.dumps
                    ({
                        "user_id": data["user_id"],
                        "interaction_data": structured_data["interaction_data"],
                    }),
                    metadata={"phase":"interaction"}
                )
                await self.send(next_msg)
            
             
        async def _build_analysis_prompt(self, raw_query, profile):
            """Construye el prompt para el análisis de personalidad y expansión de query"""
            return f"""
                Eres un experto en analizar el estilo de comunicación y preferencias de los usuarios, 
                y en mejorar consultas para búsqueda de información. Tu tarea es:

            1. ANALIZAR ESTILO DE COMUNICACIÓN:
               - Usando el mensaje del usuario, deduce sus preferencias de comunicación
               - Considera el perfil existente como contexto: {json.dumps(profile['preferences'])}
               - Parámetros a deducir:
                 * humor_score: 0.0-1.0 (0=serio, 1=divertido)
                 * formality_level: 0.0-1.0 (0=informal, 1=formal)
                 * response_style: ['direct', 'narrative', 'technical', 'socratic']
                 * preferred_topics: temas implícitos en el mensaje
                 * language: idioma detectado

            2. EXPANDIR Y RECTIFICAR LA QUERY:
               - Mejora la consulta para búsqueda en base de conocimiento
               - Expande abreviaturas, aclara términos ambiguos
               - Añade contexto histórico relevante
               - Mantén la intención original

            FORMATO DE RESPUESTA (SOLO JSON):
            {{
              "interaction_data": {{
                "humor_score": float,
                "formality_level": float,
                "response_style": string,
                "preferred_topics": [string],
                "language": string,
                "inferred_interests": [string]
              }},
              "expanded_query": string
            }}

            MENSAJE DEL USUARIO: "{raw_query}"
                        """

        async def _parse_llm_response(self, llm_response,raw_query):
            """Extrae la parte estructurada de la respuesta del LLM"""
            try:
                # Buscar el inicio del JSON
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                json_str = llm_response[start:end]
                return json.loads(json_str)
            except:
                # Fallback en caso de error
                return {
                    "interaction_data": {},
                    "expanded_query": raw_query
                }
        async def call_endpoint(self,body):
            headers = {
                "Authorization": f"Bearer sk-0iV3NeVrYrdTYJsyEeygFZ3DyatEv6F9q6v8GDdC52KDADbZ",
                "Content-Type": "application/json"
            }
            response = requests.post("https://apigateway.avangenio.net/chat/completions", headers=headers, json=body)
            response.raise_for_status()  # Lanza una excepción si hay algún error
            return response.json()
        async def send_request(self,prompt):
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
            response= await self.call_endpoint(body)
            content= response['choices'][0]['message']['content']
            return content
    async def setup(self):
        Template_Behaviour= Template()
        Template_Behaviour.metadata= {"phase":"analyzer"}
        self.add_behaviour(self.SearchProfileBehaviour(),Template_Behaviour)
        Template_Behaviour= Template()
        Template_Behaviour.metadata= {"phase":"profile"}
        self.add_behaviour(self.AnalyzerBahaviour(),Template_Behaviour)