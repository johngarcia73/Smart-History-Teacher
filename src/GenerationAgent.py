from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
import numpy as np
import faiss

class GenerationAgent(Agent):
    class GenerateBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                query, context = eval(msg.body)
                print(f"Generando respuesta para: {query}")
                
                # 1. Generar respuesta aumentada
                answer = self.agent.generate(query, context)
                
                # 2. Enviar al módulo de interfaz
                reply = Message(to="interfaz@your_xmpp_server.com")
                reply.body = answer
                await self.send(reply)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = pipeline("text-generation", model="gpt2")

        def generate(self, query: str, context: list) -> str:
            input_text = f"Contexto: {'. '.join(context)}\nPregunta: {query}\nRespuesta:"
            return self.model(input_text, max_length=500)[0]["generated_text"]