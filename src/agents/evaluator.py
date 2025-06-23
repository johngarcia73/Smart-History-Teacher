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
from utils.constants import METADATA_FILE, PROMPT_JID, SCRAPER_JID, CONFIDENCE_THRESHOLD, CRAWLER_JID
from utils.helpers import safe_json_dumps
import logging
import spacy

logger = logging.getLogger(__name__)
max_list = []

class EvaluationAgent(Agent):
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.pending_queries = {}
        self.current_query = ""
        
    async def setup(self):
        with open(METADATA_FILE, "rb") as f:
            self.metadata = pickle.load(f)
        
        self.chunks = [entry['text'] for entry in self.metadata]
        self.tokenized_chunks = [word_tokenize(chunk.lower(), language='spanish') for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        self.query_analyzer = QueryAnalyzer()
        self.score_normalizer = ScoreNormalizer()
        
        eval_template = Template(metadata={"phase": "evaluation"})
        scrape_template = Template(metadata={"phase": "scrape_result"})
        
        self.add_behaviour(self.EvaluationBehaviour(), eval_template)
        self.add_behaviour(self.ScraperResponseBehaviour(), scrape_template)
        
        logger.info(f"{self.jid} iniciado correctamente")

    class EvaluationBehaviour(CyclicBehaviour):
        async def trigger_scraping(self, query, candidates, original_sender):
            max_score = max(c["final_score"] for c in candidates) if candidates else 0
            
            if max_score < CONFIDENCE_THRESHOLD:
                logger.info(f"EvaluationAgent: Confianza baja ({max_score:.2f}), solicitando scraping")
                
            self.agent.pending_queries[query] = {
                "candidates": candidates,
                "timestamp": time.time(),
                "sender": str(original_sender),
                "query_tokens": word_tokenize(query.lower(), language='spanish')
            }
            
            scrape_msg = Message(to=CRAWLER_JID)
            scrape_msg.set_metadata("phase", "scrape_request")
            scrape_msg.body = safe_json_dumps({
                "query": query,
                "max_chunks": 10
            })
            await self.send(scrape_msg)

        async def send_to_prompt(self, query, candidates, original_sender):
            sorted_candidates = sorted(candidates, key=lambda c: c["final_score"], reverse=True)[:10]
            context = " ".join(c["text"] for c in sorted_candidates)
            
            sources = "local"
            if any(c.get("source") == "internet" for c in sorted_candidates):
                sources = "local+internet"
            
            msg = Message(to=PROMPT_JID)
            msg.set_metadata("phase", "prompt")
            msg.set_metadata("original_sender", str(original_sender))
            msg.body = safe_json_dumps({
                "query": query,
                "context": context,
                "sources": sources
            })
            await self.send(msg)
            logger.info(f"EvaluationAgent: Contexto enviado a PromptAgent para '{query}'")

        async def run(self):
            msg = await self.receive(timeout=10)
            if not msg:
                return

            try:
                data = json.loads(msg.body)
                query = data.get("query", "")
                self.agent.current_query = query
                candidates = data.get("candidates", [])
                original_sender = msg.sender

                query_tokens = word_tokenize(query.lower(), language='spanish')
                query_type = self.agent.query_analyzer.analyze(query)
                logger.info(f"EvaluationAgent: Tipo de consulta '{query_type}' para: '{query}'")

                # Normalización de puntajes FAISS
                faiss_raw_scores = [1.0 / (1.0 + c["distance"]) for c in candidates] if candidates else []
                
                faiss_norm = []
                if faiss_raw_scores:
                    q1, q3 = np.percentile(faiss_raw_scores, [25, 75])
                    iqr = q3 - q1
                    mean = np.mean(faiss_raw_scores)
                    
                    faiss_norm = [(s - mean) / iqr if iqr > 1e-6 else s for s in faiss_raw_scores]
                    # Normalizar a rango [0,1]
                    min_val = min(faiss_norm)
                    max_val = max(faiss_norm)
                    range_val = max_val - min_val if max_val - min_val > 1e-6 else 1
                    faiss_norm = [(s - min_val) / range_val for s in faiss_norm]
                    
                    
                #Sistema de bonificación
                candidate_bm25_scores = []
                if candidates:
                    bm25_full_scores = self.agent.bm25.get_scores(query_tokens)
                    
                    for i, candidate in enumerate(candidates):
                        candidate_index = candidate.get("id", i)
                        score = bm25_full_scores[candidate_index] if candidate_index < len(bm25_full_scores) else 0
                        
                        candidate_bm25_scores.append(score)
                    
                    # Normalización BM25
                    bm25_norm = self.agent.score_normalizer.robust_scale(candidate_bm25_scores)
                else:
                    bm25_norm = []
                    
                print(f"Las normas faiss son: {faiss_norm} y las bm25 son: {bm25_norm}")

                ranked_candidates = []
                for i, candidate in enumerate(candidates):
                    weights = self.agent.get_adaptive_weights(query_type, candidate, query_tokens)
                    
                    faiss_val = faiss_norm[i] if faiss_norm and i < len(faiss_norm) else 0
                    bm25_val = bm25_norm[i] if bm25_norm and i < len(bm25_norm) else 0
                    
                    print(f"El valor faiss es: {faiss_val} y el de bm25 es: {bm25_val}")
                    
                    print(candidate)
                    final_score = (
                        weights["faiss"]  * faiss_val +
                        weights["bm25"] * bm25_val
                    )
                    ranked_candidates.append({
                        "id": candidate.get("id", i),
                        "text": candidate["text"],
                        "final_score": final_score,
                        "faiss": faiss_val,
                        "bm25": bm25_val,
                        "weights": weights
                    })

                if ranked_candidates:
                    ranked_candidates.sort(key=lambda x: x["final_score"], reverse=True)
                    max_score = max(c["final_score"] for c in ranked_candidates)
                    max_list.append(max_score)
                    logger.info(f"EvaluationBehaviour: max final_score = {max_score:.2f}")

                    if max_score < CONFIDENCE_THRESHOLD:
                        await self.trigger_scraping(query, ranked_candidates, original_sender)
                    else:
                        await self.send_to_prompt(query, ranked_candidates, original_sender)
                else:
                    logger.warning("No hay candidatos para evaluar")
                
            except Exception as e:
                logger.warning(f"EvaluationBehaviour ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
            
    class ScraperResponseBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=30)
            if not msg:
                return
                
            logger.info("EvaluationAgent: Respuesta de scraper recibida")
            try:
                data = json.loads(msg.body)
                query = data["query"]
                scraped_chunks = data["scraped_data"]
                
                if query not in self.agent.pending_queries:
                    logger.warning(f"ScraperResponse: Consulta '{query}' no encontrada en pendientes")
                    return
                    
                state = self.agent.pending_queries.pop(query)
                candidates = state["candidates"]
                original_sender = state["sender"]
                query_tokens = state["query_tokens"]
                
                logger.info(f"ScraperResponse: Recibidos {len(scraped_chunks)} chunks para '{query}'")
                
                # Calcular BM25 para los nuevos chunks
                outer_scores = []
                if scraped_chunks:
                    tokenized_outer = [word_tokenize(chunk.lower(), language='spanish') for chunk in scraped_chunks]
                    outer_bm25 = BM25Okapi(tokenized_outer)
                    outer_scores = outer_bm25.get_scores(query_tokens)
                    outer_norm = self.agent.score_normalizer.sigmoid_scale(outer_scores, a=10)
                else:
                    outer_norm = []

                combined_candidates = candidates[:]
                
                for i, chunk in enumerate(scraped_chunks):
                    outer_score = outer_norm[i] if i < len(outer_norm) else 0
                    
                    trust_factor = 0.8
                    #max_local = max(c["final_score"] for c in candidates) if candidates else 1.0
                    adjusted_score = outer_score * trust_factor
                    
                    combined_candidates.append({
                        "text": chunk,
                        "source": "internet",
                        "final_score": adjusted_score
                    })
                
                # Re-rankear
                combined_candidates.sort(key=lambda x: x["final_score"], reverse=True)
                sorted_candidates = combined_candidates[:10]
                context = " ".join(c["text"] for c in sorted_candidates)
                
                prompt_msg = Message(to=PROMPT_JID)
                prompt_msg.set_metadata("phase", "prompt")
                prompt_msg.set_metadata("original_sender", original_sender)
                prompt_msg.body = safe_json_dumps({
                    "query": query,
                    "context": context,
                    "sources": "local+outerpedia"
                })
                await self.send(prompt_msg)
                logger.info(f"EvaluationAgent: Contexto mejorado enviado a PromptAgent para '{query}'")
                
            except Exception as e:
                logger.warning(f"ScraperResponse ERROR: {str(e)}")
                import traceback
                traceback.print_exc()

    def get_adaptive_weights(self, query_type, candidate, query_tokens):
        base_weights = {
            "factual": {"faiss": 0.6, "bm25": 0.4},
            "conceptual": {"faiss": 0.7, "bm25": 0.3},
            "procedural": {"faiss": 0.5, "bm25": 0.5},
            "default": {"faiss": 0.5, "bm25": 0.5}
        }

        weights = base_weights.get(query_type, {"faiss": 0.3, "bm25": 0.7}).copy()

        candidate_tokens = candidate["text"].lower().split()
        if not query_tokens:
            overlap_ratio = 0
        else:
            matching = sum(1 for token in query_tokens if token in candidate_tokens)
            overlap_ratio = matching / len(query_tokens)

        
        total = weights["faiss"] + weights["bm25"]
        normalized_weights = {
            "faiss": weights["faiss"] / total,
            "bm25": weights["bm25"] / total
        }
        return normalized_weights


class QueryAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except:
            raise ImportError("Modelo de spaCy para español no instalado.")
    
    def analyze(self, query):
        doc = self.nlp(query)
        question = query.lower()
        
        # Detección de tipo de pregunta
        question_types = {
            "factual": ["quién", "cuándo", "dónde", "cuántos", "cuántas", "en qué año"],
            "procedural": ["cómo", "pasos", "proceso", "método"],
            "conceptual": ["por qué", "causas", "consecuencias", "impacto", "efectos", "explica", "diferencia"]
        }
        
        for q_type, markers in question_types.items():
            if any(marker in question for marker in markers):
                return q_type
        
        # Predeterminado conceptual para preguntas complejas
        return "conceptual"


class ScoreNormalizer:
    def sigmoid_scale(self, scores, a=10, b=None):
        if scores is None or len(scores) == 0:
            return []
            
        scores = np.array(scores)
        if b is None:
            b = np.percentile(scores, 75)
        
        scores = np.clip(scores, -500, 500)
        return list(1 / (1 + np.exp(-a * (scores - b))))
    
    def robust_scale(self, scores):
        """Escalado usando IQR con normalización a [0,1]"""
        if scores is None or len(scores) == 0:
            return []
            
        scores = np.array(scores)
        if len(scores) < 4:
            return self.minmax_scale(scores)
            
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        
        if iqr < 1e-6:
            return self.minmax_scale(scores)
            
        median = np.median(scores)
        scaled = (scores - median) / iqr
        
        # Normalizar a [0,1]
        min_val = np.min(scaled)
        max_val = np.max(scaled)
        range_val = max_val - min_val
        if range_val < 1e-6:
            return [0.5] * len(scores)
            
        return list((scaled - min_val) / range_val)
    
    def minmax_scale(self, scores):
        if scores is None or len(scores) == 0:
            return []
            
        scores = np.array(scores)
        min_val = np.min(scores)
        max_val = np.max(scores)
        range_val = max_val - min_val
        if range_val < 1e-6:
            return [0.5] * len(scores)
        return list((scores - min_val) / range_val)