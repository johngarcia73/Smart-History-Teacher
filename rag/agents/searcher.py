import os
import pickle
import faiss
import numpy as np
import json
import asyncio
from nltk.tokenize import word_tokenize
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from utils.constants import INDEX_FILE, METADATA_FILE, DOCUMENTS_FOLDER, EVAL_JID
from utils.helpers import safe_json_dumps
from indexer import build_index
from fastembed import TextEmbedding


class SearchAgent(Agent):
    async def setup(self):
        print("DistributedSearchAgent: Verificando base de conocimientos...")
        if not os.path.exists(DOCUMENTS_FOLDER):
            os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
            
        if not (os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE)):
            print(f"Construyendo base desde {DOCUMENTS_FOLDER}...")
            build_index(DOCUMENTS_FOLDER, INDEX_FILE, METADATA_FILE)

        self.index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            self.metadata = pickle.load(f)
        self.chunks = [entry['text'] for entry in self.metadata]
        self.tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        #self.embedder =  TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", cache_dir="model_cache")

        
        self.add_behaviour(self.SearchBehaviour())
        print(f"{self.jid} iniciado correctamente")

    class SearchBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg and msg.get_metadata("phase") == "query":
                payload = json.loads(msg.body)
                query = payload.get("query", "")
                print(f"DistributedSearchAgent: Consulta recibida: {query}")
                
                query_embedding = self.agent.embedder.encode([query])[0]
                query_embedding = np.array([query_embedding])
                distances, indices = self.agent.index.search(query_embedding, 10)
                
                candidates = []
                for dist, idx in zip(distances[0], indices[0]):
                    candidates.append({
                        "id": int(idx),
                        "text": self.agent.metadata[idx]['text'],
                        "distance": float(dist)
                    })
                
                new_msg = Message(to=EVAL_JID)
                new_msg.set_metadata("phase", "evaluation")
                new_msg.body = safe_json_dumps({"query": query, "candidates": candidates})
                await self.send(new_msg)
                print("DistributedSearchAgent: Resultados enviados para evaluación")
            else:
                await asyncio.sleep(1)