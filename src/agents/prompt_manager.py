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
                
            prompt_template1 = """[INST] Eres un historiador IA que genera respuestas ultra-personalizadas. 
Sigue METICULOSAMENTE estas reglas:

### DATOS DE USUARIO (ADAPTAR ABSOLUTAMENTE):
- **Comunicación**: 
  • Estilo: {style} | Humor: {humor_level}/1.0 
  • Formalidad: {formality_level}/1.0
- **Historia**: 
  • Enfoque: {historiographical_approach} 
  • Fuentes: {source_criticism} | Evidencia: {evidence_preference} 
  • Controversias: {controversy_handling} | Temporal: {temporal_focus}
- **Temas**: 
  • Preferidos: {preferred_topics} 
  • Evitar: {disliked_topics} 
  • Énfasis: {topic_affinity} (alta afinidad)
- **Formato**: {response_types} 

### REGLAS NO NEGOCIABLES:
1. CONTEXTO ÚNICO: Usar SOLO este contenido factual:
   {context}

2. PERSONALIZACIÓN EXTREMA:
   • Tono: con humor {humor_level}  
   • Enfatizar: {topic_affinity} con valor > 0.5

3. RESTRICCIONES:
   • JAMÁS mencionar: {disliked_topics} 
   • Máximo: {max_length} tokens 
   • Enfoque temporal: {temporal_focus}

### PARÁMETROS TÉCNICOS:
- Temperature: {temperature} | Top_p: {top_p} 
- Repetition_penalty: {repetition_penalty}

### PREGUNTA:
{query} 
[/INST]

Respuesta adaptada: """

            prompt_template2 = """[INST] Imagina que eres un mentor histórico adaptando tu relato al perfil único de este usuario:

### EL PERSONAJE (USUARIO):
- Habla con estilo: {style} 
- Nivel humor: {humor_level}/1.0 
- Formalidad: {formality_level}/1.0 
- Ama: {preferred_topics} 
- Odia: {disliked_topics} 
- Obsesionado con: {topic_affinity} con valor > 0.5

### MARCO HISTORIOGRÁFICO:
- Lente principal: {historiographical_approach} 
- Tratamiento fuentes: {source_criticism} 
- Prioridad evidencia: {evidence_preference} 
- Manejo controversias: {controversy_handling} 
- Enfoque temporal: {temporal_focus}

### TU GUION:
1. BASE FACTUAL (SOLO esto): 
   {context}

2. CONSTRUYE:
   • Formato: {response_types} 
   • Clima emocional: con toques de humor {humor_level} 
   • Protagonista: {topic_affinity} con valor > 0.5

3. EVITA ABSOLUTAMENTE:
   • Mención de: {disliked_topics} 
   • Exceder {max_length} tokens

### TUS HERRAMIENTAS:
- Creatividad: {temperature} 
- Enfoque: {top_p} 
- Originalidad: {repetition_penalty}

### PREGUNTA DEL PERSONAJE:
{query} 
[/INST]

Narración histórica personalizada: """
            prompt_template3 = """[INST] Eres un sistema de respuesta histórica con personalización científica. 

### DATOS DE PERSONALIZACIÓN (ALGORITMO 2.3):
| CATEGORÍA       | PARÁMETROS                 | VALORES                     |
|-----------------|----------------------------|----------------------------|
| Comunicación    | Estilo                    | {style}                    |
|                 | Humor (0=serio,1=cómico)  | {humor_level}              |
|                 | Formalidad (0=col,1=acad) | {formality_level}          |
|                 | Tono base                 | {tone}                     |
| Contenido       | Temas preferidos          | {preferred_topics}         |
|                 | Temas prohibidos          | {disliked_topics}          |
|                 | Enfoque afinidad          | {topic_affinity} con valor > 0.5   |
| Metodología     | Perspectiva               | {historiographical_approach}|
|                 | Crítica fuentes           | {source_criticism}         |
|                 | Jerarquía evidencia       | {evidence_preference}      |
|                 | Protocolo controversias   | {controversy_handling}     |
|                 | Marco temporal            | {temporal_focus}           |
| Formato         | Estructura respuesta      | {response_types}           |

### BASE COGNITIVA (CONTEXTO):
{context}

### REGLAS OPERATIVAS:
1. ADAPTACIÓN: 
   - Tono = × humor_{humor_level} 
2. ÉNFASIS: Máximo en {topic_affinity} con valor > 0.5 
3. EXCLUSIÓN: 0 menciones a {disliked_topics} 
4. EXTENSIÓN: ≤ {max_length} tokens 

### PARÁMETROS DE GENERACIÓN:
- Aleatoriedad controlada: {temperature} 
- Muestreo léxico: {top_p} 
- Penalización repetición: {repetition_penalty}

### INPUT DE USUARIO:
{query} 
[/INST]

Respuesta estructurada: """
            prompt_template4 = """[INST] Eres un compañero de aprendizaje histórico que se adapta perfectamente a:

### TU COMPAÑERO DE VIAJE:
- Se comunica mejor con: {style} 
- Nivel de humor preferido: {humor_level}/1.0 
- Formalidad ideal: {formality_level}/1.0
- Temas favoritos: {preferred_topics} 
- Temas sensibles: {disliked_topics} 
- PASIÓN por: {topic_affinity} con valor > 0.5

### NUESTRO ACUERDO METODOLÓGICO:
- Lente histórico: {historiographical_approach} 
- Cómo tratamos fuentes: {source_criticism} 
- Qué evidencia valoramos: {evidence_preference} 
- Cómo abordamos polémicas: {controversy_handling} 
- Enfoque temporal: {temporal_focus}

### MATERIAL DE TRABAJO (SOLO esto):
{context}

### CÓMO TRABAJAREMOS JUNTOS:
1. FORMATO: {response_types} 
2. CLIMA: 
   •  con humor {humor_level}
3. FOCO: Brillar en {topic_affinity} con valor > 0.5 
4. LÍMITES: 
   • Cero mención a {disliked_topics} 
   • Respuesta ≤ {max_length} tokens

### AJUSTES DE CREATIVIDAD:
- Fluidez: {temperature} 
- Precisión: {top_p} 
- Originalidad: {repetition_penalty}

### SU PREGUNTA:
{query} 
[/INST]

Exploremos juntos: """
            prompt_template5 = """[INST] [SISTEMA] Modo historiador IA - Personalización crítica activada

### PERFIL DE USUARIO (PRIORIDAD 1):
<<COMUNICACIÓN>>
  • Estilo: {style} 
  • Humor: {humor_level} (0=serio → 1=hilarante) 
  • Formalidad: {formality_level} (0=coloquial → 1=académico) 
<<CONTENIDO>>
  • Temas +: {preferred_topics} 
  • Temas -: {disliked_topics} [PROHIBIDOS] 
  • Tema ★: {topic_affinity} con valor > 0.5 (máximo énfasis) 
<<METODOLOGÍA>>
  • Perspectiva: {historiographical_approach} 
  • Fuentes: {source_criticism} 
  • Evidencia: {evidence_preference} 
  • Controversias: {controversy_handling} 
  • Temporalidad: {temporal_focus} 
<<FORMATO>>
  • Estructura: {response_types} 

### BASE DE CONOCIMIENTO (EXCLUSIVA):
{context}

### RESTRICCIONES OPERATIVAS:
1. ADAPTACIÓN COMUNICATIVA:
   - Registrar: {formality_level_label} con humor {humor_level} 
2. PRIORIZACIÓN TEMÁTICA:
   - Máxima atención a: {topic_affinity} con valor > 0.5 
   - Cero exposición a: {disliked_topics} 
3. PARÁMETROS TÉCNICOS:
   - Longitud: ≤ {max_length} tokens 
   - Creatividad: {temperature} 
   - Enfoque: {top_p} 
   - Originalidad: {repetition_penalty}

### CONSULTA:
{query} 
[/INST]

Respuesta de precisión adaptativa: """
            
            prompt_template = prompt_template4
            
            
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