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
                [INST] You are a history expert assistant who answers questions in a highly personalized manner. 
                Strictly follow these guidelines:
                ### User profile and preferences:
                - **Communication style**: {style}
                - **Humor level**: {humor_level} (0: serious, 1: humorous)
                - **Formality**: {formality_level} (0: colloquial, 1: academic)
                - **Preferred topics**: {preferred_topics}
                - **Avoided topics**: {disliked_topics}
                - **Preferred response types**: {response_types}

                ### Required historical focus:
                - **Historiographical perspective**: {historiographical_approach}
                - **Source treatment**: {source_criticism}
                - **Evidence preference**: {evidence_preference}
                - **Controversy handling**: {controversy_handling}
                - **Temporal focus**: {temporal_focus}

                ### Key instructions:
                1. Answer based ONLY on the provided context
                2. Adapt tone according to humor_level and formality_level
                3. Use {response_types} as main format
                4. Emphasize topics with high affinity ({topic_affinity})
                5. Avoid mentioning {disliked_topics}
                6. Apply {historiographical_approach} when analyzing events
                7. Handle controversies with {controversy_handling}
                8. Limit response to {max_length} tokens

                ### Context (factual base):
                {context}

                ### User question:
                {query}

                ### Technical generation parameters:
                - Controlled creativity (temperature: {temperature})
                - Lexical focus (top_p: {top_p})
                - Repetition prevention (repetition_penalty: {repetition_penalty})
                [/INST]

                Personalized response:"""
                
            prompt_template1 = """[INST] You are an AI historian who generates ultra-personalized responses. 
Follow these rules METICULOUSLY:

### USER DATA (ABSOLUTELY ADAPT):
- **Communication**: 
  • Style: {style} | Humor: {humor_level}/1.0 
  • Formality: {formality_level}/1.0
- **History**: 
  • Focus: {historiographical_approach} 
  • Sources: {source_criticism} | Evidence: {evidence_preference} 
  • Controversies: {controversy_handling} | Temporal: {temporal_focus}
- **Topics**: 
  • Preferred: {preferred_topics} 
  • Avoid: {disliked_topics} 
  • Emphasis: {topic_affinity} (high affinity)
- **Format**: {response_types} 

### NON-NEGOTIABLE RULES:
1. UNIQUE CONTEXT: Use ONLY this factual content:
   {context}

2. EXTREME PERSONALIZATION:
   • Tone: with humor level {humor_level}  
   • Emphasize: {highest_affinity_topic}

3. RESTRICTIONS:
   • NEVER mention: {disliked_topics} 
   • Maximum: {max_length} tokens 
   • Temporal focus: {temporal_focus}

### TECHNICAL PARAMETERS:
- Temperature: {temperature} | Top_p: {top_p} 
- Repetition_penalty: {repetition_penalty}

### QUESTION:
{query} 
[/INST]

Adapted response: """

            prompt_template2 = """[INST] Imagine you are a historical mentor adapting your narrative to this user's unique profile:

### THE CHARACTER (USER):
- Speaks with style: {style} 
- Humor level: {humor_level}/1.0 
- Formality: {formality_level}/1.0 
- Loves: {preferred_topics} 
- Hates: {disliked_topics} 
- Obsessed with: {highest_affinity_topic}

### HISTORIOGRAPHICAL FRAMEWORK:
- Main lens: {historiographical_approach} 
- Source treatment: {source_criticism} 
- Evidence priority: {evidence_preference} 
- Controversy handling: {controversy_handling} 
- Temporal focus: {temporal_focus}

### YOUR SCRIPT:
1. FACTUAL BASE (ONLY this): 
   {context}

2. BUILD:
   • Format: {response_types} 
   • Emotional climate: with touches of humor level {humor_level} 
   • Protagonist: {highest_affinity_topic}

3. ABSOLUTELY AVOID:
   • Mention of: {disliked_topics} 
   • Exceeding {max_length} tokens

### YOUR TOOLS:
- Creativity: {temperature} 
- Focus: {top_p} 
- Originality: {repetition_penalty}

### CHARACTER'S QUESTION:
{query} 
[/INST]

Personalized historical narrative: """
            prompt_template3 = """[INST] You are a historical response system with scientific personalization. 

### PERSONALIZATION DATA (ALGORITHM 2.3):
| CATEGORY         | PARAMETERS                 | VALUES                     |
|------------------|----------------------------|----------------------------|
| Communication    | Style                     | {style}                    |
|                  | Humor (0=serious,1=comic) | {humor_level}              |
|                  | Formality (0=coll,1=acad) | {formality_level}          |
|                  | Base tone                 | {tone}                     |
| Content          | Preferred topics          | {preferred_topics}         |
|                  | Forbidden topics          | {disliked_topics}          |
|                  | Affinity focus            | {highest_affinity_topic}   |
| Methodology      | Perspective               | {historiographical_approach}|
|                  | Source criticism          | {source_criticism}         |
|                  | Evidence hierarchy        | {evidence_preference}      |
|                  | Controversy protocol      | {controversy_handling}     |
|                  | Temporal framework        | {temporal_focus}           |
| Format           | Response structure        | {response_types}           |

### COGNITIVE BASE (CONTEXT):
{context}

### OPERATIONAL RULES:
1. ADAPTATION: 
   - Tone = × humor_{humor_level} 
2. EMPHASIS: Maximum on {highest_affinity_topic} 
3. EXCLUSION: 0 mentions of {disliked_topics} 
4. LENGTH: ≤ {max_length} tokens 

### GENERATION PARAMETERS:
- Controlled randomness: {temperature} 
- Lexical sampling: {top_p} 
- Repetition penalty: {repetition_penalty}

### USER INPUT:
{query} 
[/INST]

Structured response: """
            prompt_template4 = """[INST] You are a historical learning companion that perfectly adapts to:

### YOUR TRAVEL COMPANION:
- Communicates best with: {style} 
- Preferred humor level: {humor_level}/1.0 
- Ideal formality: {formality_level}/1.0
- Favorite topics: {preferred_topics} 
- Sensitive topics: {disliked_topics} 
- PASSION for: {highest_affinity_topic}

### OUR METHODOLOGICAL AGREEMENT:
- Historical lens: {historiographical_approach} 
- How we treat sources: {source_criticism} 
- What evidence we value: {evidence_preference} 
- How we approach controversies: {controversy_handling} 
- Temporal focus: {temporal_focus}

### WORK MATERIAL (ONLY this):
{context}

### HOW WE WILL WORK TOGETHER:
1. FORMAT: {response_types} 
2. CLIMATE: 
   • {tone} tone with humor level {humor_level}
3. FOCUS: Shine on {highest_affinity_topic} 
4. LIMITS: 
   • Zero mention of {disliked_topics} 
   • Response ≤ {max_length} tokens

### CREATIVITY SETTINGS:
- Fluency: {temperature} 
- Precision: {top_p} 
- Originality: {repetition_penalty}

### YOUR QUESTION:
{query} 
[/INST]

Let's explore together: """
            prompt_template5 = """[INST] [SYSTEM] AI historian mode - Critical personalization activated

### USER PROFILE (PRIORITY 1):
<<COMMUNICATION>>
  • Style: {style} 
  • Humor: {humor_level} (0=serious → 1=hilarious) 
  • Formality: {formality_level} (0=colloquial → 1=academic) 
<<CONTENT>>
  • Topics +: {preferred_topics} 
  • Topics -: {disliked_topics} [FORBIDDEN] 
  • Topic ★: {highest_affinity_topic} (maximum emphasis) 
<<METHODOLOGY>>
  • Perspective: {historiographical_approach} 
  • Sources: {source_criticism} 
  • Evidence: {evidence_preference} 
  • Controversies: {controversy_handling} 
  • Temporality: {temporal_focus} 
<<FORMAT>>
  • Structure: {response_types} 

### KNOWLEDGE BASE (EXCLUSIVE):
{context}

### OPERATIONAL CONSTRAINTS:
1. COMMUNICATIVE ADAPTATION:
   - Register: {formality_level_label} with humor {humor_level} 
2. THEMATIC PRIORITIZATION:
   - Maximum attention to: {highest_affinity_topic} 
   - Zero exposure to: {disliked_topics} 
3. TECHNICAL PARAMETERS:
   - Length: ≤ {max_length} tokens 
   - Creativity: {temperature} 
   - Focus: {top_p} 
   - Originality: {repetition_penalty}

### QUERY:
{query} 
[/INST]

Adaptive precision response: """
            
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