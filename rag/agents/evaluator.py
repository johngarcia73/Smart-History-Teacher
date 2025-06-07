import json
import asyncio
import pickle
from nltk.tokenize import word_tokenize
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from rank_bm25 import BM25Okapi
from utils.constants import METADATA_FILE, PROMPT_JID
from utils.helpers import safe_json_dumps

class EvaluationAgent(Agent):
    async def setup(self):
        with open(METADATA_FILE, "rb") as f:
            self.metadata = pickle.load(f)
        self.chunks = [entry['text'] for entry in self.metadata]
        self.tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.add_behaviour(self.EvaluationBehaviour())
        print(f"{self.jid} iniciado correctamente")

    class EvaluationBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg and msg.get_metadata("phase") == "evaluation":
                data = json.loads(msg.body)
                query = data.get("query", "")
                candidates = data.get("candidates", [])
                
                query_tokens = word_tokenize(query.lower())
                bm25_all = self.agent.bm25.get_scores(query_tokens)
                
                ranked_candidates = []
                for candidate in candidates:
                    faiss_score = 1.0 / (1.0 + candidate["distance"])
                    bm25_score = bm25_all[candidate["id"]]
                    final_score = 0.5 * faiss_score + 0.5 * bm25_score
                    ranked_candidates.append({
                        "text": candidate["text"],
                        "final_score": final_score
                    })
                
                ranked_candidates.sort(key=lambda x: x["final_score"], reverse=True)
                context = " ".join([entry["text"] for entry in ranked_candidates[:5]])
                
                new_msg = Message(to=PROMPT_JID)
                new_msg.set_metadata("phase", "prompt")
                new_msg.body = safe_json_dumps({"query": query, "context": context})
                await self.send(new_msg)
                print("DistributedEvaluationAgent: Contexto enviado para generación")
            else:
                await asyncio.sleep(1)