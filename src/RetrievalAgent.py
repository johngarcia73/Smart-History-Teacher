from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
import numpy as np
import faiss

class RetrievalAgent:
    def __init__(self, index_path: str):
        self.index = faiss.read_index(index_path)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def search(self, query: str, k=5) -> list:
        query_embed = self.encoder.encode(query)
        distances, indices = self.index.search(np.array([query_embed]), k)
        return [knowledge_base[i] for i in indices[0]]
    

# --------------------------------------------
# Agente de Búsqueda (Retrieval Agent)
# --------------------------------------------
class RetrievalAgent(Agent):
    class SearchBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)  # Espera mensajes
            if msg:
                print(f"Recibida consulta: {msg.body}")
                
                # 1. Procesar consulta
                results = self.agent.search(msg.body)
                
                # 2. Enviar respuesta al coordinador
                reply = Message(to=msg.sender)
                reply.body = str(results)  # Serializar resultados
                await self.send(reply)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Cargar índice FAISS durante la inicialización
            self.index = faiss.read_index("historico.index")
            self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        def search(self, query: str) -> list:
            # Convertir consulta a embedding
            query_embed = self.encoder.encode(query)
            # Buscar en FAISS
            _, indices = self.index.search(np.array([query_embed]), 5)
            return [self.knowledge_base[i] for i in indices[0]]
