import json
import requests
#from aioxmpp import JID
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from nltk.tokenize import sent_tokenize
from utils.constants import WIKIPEDIA_API

class ScraperAgent(Agent):
    async def setup(self):
        self.add_behaviour(self.ScrapeBehaviour())
        print(f"{self.jid} iniciado correctamente")

    class ScrapeBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg and msg.get_metadata("phase") == "scrape_request":
                try:
                    data = json.loads(msg.body)
                    query = data["query"]
                    max_chunks = data.get("max_chunks", 3)
                    
                    print(f"ScraperAgent: Buscando en Wikipedia para: '{query}'")
                    scraped_data = await self.scrape_wikipedia(query, max_chunks)
                    
                    reply = msg.make_reply()
                    reply.set_metadata("phase", "scrape_result")
                    reply.body = json.dumps({
                        "query": query,
                        "scraped_data": scraped_data
                    })
                    await self.send(reply)
                    print(f"ScraperAgent: Enviados {len(scraped_data)} chunks a EvaluationAgent")
                    
                except Exception as e:
                    print(f"ScraperAgent Error: {str(e)}")

        async def scrape_wikipedia(self, query, max_chunks=3):
            """Recupera información relevante de Wikipedia en español"""
            
            
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "srprop": "size",
                "utf8": 1
            }
            
            try:
                # Búsqueda inicial de artículos
                search_response = requests.get(WIKIPEDIA_API, params=params)
                search_data = search_response.json()
                articles = search_data.get("query", {}).get("search", [])[:2]
                
                # Extracción de contenido relevante
                scraped_chunks = []
                for article in articles:
                    content_params = {
                        "action": "query",
                        "prop": "extracts",
                        "exintro": True,
                        "explaintext": True,
                        "titles": article["title"],
                        "format": "json",
                        "utf8": 1
                    }
                    content_response = requests.get(WIKIPEDIA_API, params=content_params)
                    content_data = content_response.json()
                    
                    # Procesamiento de contenido
                    pages = content_data.get("query", {}).get("pages", {})
                    for page_id, page in pages.items():
                        extract = page.get("extract", "")
                        if extract:
                            # Tokenización y chunking
                            sentences = sent_tokenize(extract, language='spanish')
                            chunks = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
                            scraped_chunks.extend(chunks[:max_chunks])
                
                return scraped_chunks[:max_chunks]
                
            except Exception as e:
                print(f"Wikipedia API Error: {str(e)}")
                return []