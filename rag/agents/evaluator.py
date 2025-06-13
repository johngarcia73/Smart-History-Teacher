import json
import asyncio
import pickle
import time
import numpy as np
from nltk.tokenize import word_tokenize
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template
from rank_bm25 import BM25Okapi
from utils.constants import METADATA_FILE, PROMPT_JID, SCORE_THRESHOLD, SCRAPER_JID, CONFIDENCE_THRESHOLD, CRAWLER_JID
from utils.helpers import safe_json_dumps
from agents.query_analyzer import QueryAnalyzer
from agents.score_normalizer import ScoreNormalizer

class EvaluationAgent(Agent):
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.pending_queries = {}  # Diccionario para múltiples consultas
        self.current_query = ""
        
    async def setup(self):
        # Cargar metadatos
        with open(METADATA_FILE, "rb") as f:
            self.metadata = pickle.load(f)
        
        # Inicializar BM25 con tokenización en español
        self.chunks = [entry['text'] for entry in self.metadata]
        self.tokenized_chunks = [word_tokenize(chunk.lower(), language='spanish') for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        self.query_analyzer = QueryAnalyzer()
        self.score_normalizer = ScoreNormalizer()
        
        eval_template = Template(metadata={"phase": "evaluation"})
        scrape_template = Template(metadata={"phase": "scrape_result"})
        
        self.add_behaviour(self.EvaluationBehaviour(), eval_template)
        self.add_behaviour(self.ScraperResponseBehaviour(), scrape_template)
        
        print(f"{self.jid} iniciado correctamente")

    class EvaluationBehaviour(CyclicBehaviour):
        async def trigger_scraping(self, query, candidates, original_sender):
            """Activa scraping cuando la confianza es baja"""
            max_score = max(c["final_score"] for c in candidates)
            if max_score < CONFIDENCE_THRESHOLD:
                print(f"EvaluationAgent: Confianza baja ({max_score:.2f}), solicitando scraping")
                
                # Guardar estado de la consulta
                self.agent.pending_queries[query] = {
                    "candidates": candidates,
                    "timestamp": time.time(),
                    "sender": str(original_sender)
                }
                
                scrape_msg = Message(to=CRAWLER_JID)
                scrape_msg.set_metadata("phase", "scrape_request")
                scrape_msg.body = safe_json_dumps({
                    "query": query,
                    "max_chunks": 10
                })
                await self.send(scrape_msg)
            else:
                await self.send_to_prompt(query, candidates, original_sender)

        async def send_to_prompt(self, query, candidates, original_sender):
            """Envía los mejores resultados al PromptAgent"""
            sorted_candidates = sorted(candidates, key=lambda c: c["final_score"], reverse=True)[:5]
            context = " ".join(c["text"] for c in sorted_candidates)
            
            # Determinar fuentes según si existen datos de scraping
            sources = "local"
            if any(c.get("source") == "wikipedia" for c in sorted_candidates):
                sources = "local+wikipedia"
            
            msg = Message(to=PROMPT_JID)
            msg.set_metadata("phase", "prompt")
            msg.set_metadata("original_sender", str(original_sender))
            msg.body = safe_json_dumps({
                "query": query,
                "context": context,
                "sources": sources
            })
            await self.send(msg)
            print(f"EvaluationAgent: Contexto enviado a PromptAgent para '{query}'")

        async def run(self):
            msg = await self.receive(timeout=10)
            if not msg:
                return

            print(f"EvaluationAgent: Mensaje recibido de {msg.sender}")
            try:
                data = json.loads(msg.body)
                query = data.get("query", "")
                self.agent.current_query = query
                candidates = data.get("candidates", [])
                
                original_sender = msg.sender

                # 1. Tokenizar la consulta (en español)
                query_tokens = word_tokenize(query.lower(), language='spanish')

                query_type = self.agent.query_analyzer.analyze(query)
                print(f"EvaluationAgent: Tipo de consulta '{query_type}' para: '{query}'")

                # Se usa la fórmula 1/(1+d)
                faiss_raw_scores = [1.0 / (1.0 + c["distance"]) for c in candidates]
                # Aplicar la normalización sigmoidea para suavizar las diferencias
                faiss_norm = self.agent.score_normalizer.sigmoid_scale(faiss_raw_scores, a=10)

                bm25_full_scores = self.agent.bm25.get_scores(query_tokens)
                candidate_bm25_scores = []
                for i, candidate in enumerate(candidates):
                    candidate_index = candidate.get("id", i)
                    candidate_bm25_scores.append(bm25_full_scores[candidate_index])
                bm25_norm = self.agent.score_normalizer.sigmoid_scale(candidate_bm25_scores, a=10)




                ranked_candidates = []
                for i, candidate in enumerate(candidates):
                    weights = self.agent.get_adaptive_weights(query_type, candidate)
                    final_score = (
                        weights["faiss"] * faiss_norm[i] +
                        weights["bm25"] * bm25_norm[i]
                    )
                    ranked_candidates.append({
                        "id": candidate.get("id", i),
                        "text": candidate["text"],
                        "final_score": final_score,
                        "faiss": faiss_norm[i],
                        "bm25": bm25_norm[i],
                        "weights": weights
                    })

                ranked_candidates.sort(key=lambda x: x["final_score"], reverse=True)
                max_score = max(c["final_score"] for c in ranked_candidates)
                print(f"EvaluationBehaviour: max final_score = {max_score:.2f}")

                if max_score < CONFIDENCE_THRESHOLD:
                    await self.trigger_scraping(query, ranked_candidates, original_sender)
                else:
                    await self.send_to_prompt(query, ranked_candidates, original_sender)
                
            except Exception as e:
                print(f"EvaluationBehaviour ERROR: {str(e)}")
                import traceback
                traceback.print_exc()

    class ScraperResponseBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=30)
            if not msg:
                return
                
            print("EvaluationAgent: Respuesta de scraper recibida")
            try:
                data = json.loads(msg.body)
                query = data["query"]
                scraped_chunks = data["scraped_data"]
                
                # Recuperar estado de la consulta
                if query not in self.agent.pending_queries:
                    print(f"ScraperResponse: Consulta '{query}' no encontrada en pendientes")
                    return
                    
                state = self.agent.pending_queries.pop(query)
                candidates = state["candidates"]
                original_sender = state["sender"]
                
                print(f"ScraperResponse: Recibidos {len(scraped_chunks)} chunks para '{query}'")
                
                # Combinar resultados locales con datos externos:
                combined_candidates = candidates[:3]  # Mejores candidatos locales
                
                # Añadir chunks de Wikipedia como nuevos candidatos
                for chunk in scraped_chunks:
                    combined_candidates.append({
                        "text": chunk,
                        "source": "wikipedia",
                        "final_score": CONFIDENCE_THRESHOLD * 0.8  # Score base
                    })
                
                # Re-ranquear combinados
                combined_candidates.sort(key=lambda x: x["final_score"], reverse=True)
                sorted_candidates = combined_candidates[:5]
                context = " ".join(c["text"] for c in sorted_candidates)
                
                # Enviar a PromptAgent
                prompt_msg = Message(to=PROMPT_JID)
                prompt_msg.set_metadata("phase", "prompt")
                prompt_msg.set_metadata("original_sender", original_sender)
                prompt_msg.body = safe_json_dumps({
                    "query": query,
                    "context": context,
                    "sources": "local+wikipedia"
                })
                await self.send(prompt_msg)
                print(f"EvaluationAgent: Contexto mejorado enviado a PromptAgent para '{query}'")
                
            except Exception as e:
                print(f"ScraperResponse ERROR: {str(e)}")
                import traceback
                traceback.print_exc()

    def get_adaptive_weights(self, query_type, candidate):
        """Asigna pesos dinámicos según tipo de consulta y características del candidato"""
        base_weights = {
            "factual": {"faiss": 0.7, "bm25": 0.3},
            "conceptual": {"faiss": 0.4, "bm25": 0.6},
            "procedural": {"faiss": 0.5, "bm25": 0.5}
        }
        
        weights = base_weights.get(query_type, {"faiss": 0.5, "bm25": 0.5}).copy()
        
        # Ajuste basado en la longitud del candidato
        text_length = len(candidate["text"])
        if text_length > 300:  # Textos largos confían más en semántica
            weights["faiss"] = min(weights["faiss"] + 0.1, 0.9)
            weights["bm25"] = max(weights["bm25"] - 0.1, 0.1)
        elif text_length < 100:  # Textos cortos confían más en términos
            weights["bm25"] = min(weights["bm25"] + 0.1, 0.9)
            weights["faiss"] = max(weights["faiss"] - 0.1, 0.1)
        
        # Normalizar para que la suma sea 1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
