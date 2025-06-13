import requests
from bs4 import BeautifulSoup
import json
import asyncio
import logging
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
import re
from urllib.parse import urlparse, parse_qs, quote, unquote

# Configuración del logger
logger = logging.getLogger("CrawlerAgentLogger")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

#############################################
# Búsqueda y Scraping Ultra-Simplificado (CORREGIDO)
#############################################

def simple_search(query, limit=5):
    """Búsqueda directa con extracción precisa"""
    logger.info(f"Buscando: '{query}'")
    try:
        # URL de búsqueda
        query_encoded = quote(query)
        url = f"https://duckduckgo.com/html/?q={query_encoded}"
        
        # Headers para parecer navegador real
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9",
            "Referer": "https://duckduckgo.com/",
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parsear HTML con BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Encontrar todos los enlaces de resultados
        urls = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Manejar URLs de DuckDuckGo
            if href.startswith('/l/?uddg=') or 'uddg=' in href:
                # Extraer parámetro uddg
                parsed = urlparse(href)
                query_params = parse_qs(parsed.query)
                if 'uddg' in query_params:
                    real_url = query_params['uddg'][0]
                    urls.append(real_url)
                else:
                    # Buscar manualmente si falla el parseo
                    start = href.find('uddg=') + 5
                    end = href.find('&', start)
                    if end == -1:
                        real_url = href[start:]
                    else:
                        real_url = href[start:end]
                    urls.append(unquote(real_url))
            
            # Agregar URLs directas
            elif href.startswith('http') and 'duckduckgo' not in href:
                urls.append(href)
                
            # Limitar resultados
            if len(urls) >= limit:
                break
        
        return urls[:limit]
    
    except Exception as e:
        logger.error(f"Error en búsqueda: {str(e)}")
        return []

def simple_scrape(url, max_chunks=3):
    """Extracción de texto crudo sin complicaciones"""
    try:
        logger.info(f"Scrapeando: {url}")
        
        # Obtener HTML
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        # Extraer texto simple
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Eliminar elementos no deseados
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        
        # Obtener todo el texto
        text = soup.get_text(separator=' ', strip=True)
        
        # Limpiar y dividir en chunks de 500 caracteres
        chunks = []
        current_chunk = ""
        for char in text:
            current_chunk += char
            if len(current_chunk) >= 500 and char in '.!?':
                chunks.append(current_chunk.strip())
                current_chunk = ""
                if len(chunks) >= max_chunks:
                    break
        
        if current_chunk and len(chunks) < max_chunks:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error scrapeando {url}: {str(e)}")
        return []

async def simple_scrape_web(query, max_chunks=5):
    """Proceso completo simplificado"""
    logger.info(f"Iniciando scraping para: '{query}'")
    
    # Paso 1: Buscar URLs
    urls = simple_search(query)
    logger.info(f"Encontradas {len(urls)} URLs")
    
    # Paso 2: Scrapear cada URL
    all_chunks = []
    for url in urls:
        chunks = simple_scrape(url, max_chunks - len(all_chunks))
        if chunks:
            all_chunks.extend(chunks)
            logger.info(f"Obtenidos {len(chunks)} chunks de {url}")
        
        if len(all_chunks) >= max_chunks:
            break
    
    logger.info(f"Total chunks obtenidos: {len(all_chunks)}")
    return all_chunks[:max_chunks]

#############################################
# Agente CrawlerAgent
#############################################

class CrawlerAgent(Agent):
    class CrawlBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=30)
            if msg and msg.get_metadata("phase") == "scrape_request":
                try:
                    data = json.loads(msg.body)
                    query = data["query"]
                    max_chunks = data.get("max_chunks", 5)
                    
                    logger.info(f"Solicitud recibida para: '{query}'")
                    scraped_data = await simple_scrape_web(query, max_chunks)
                    
                    reply = msg.make_reply()
                    reply.set_metadata("phase", "scrape_result")
                    reply.body = json.dumps({
                        "query": query,
                        "scraped_data": scraped_data
                    })
                    await self.send(reply)
                    logger.info(f"Enviados {len(scraped_data)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error en el procesamiento: {str(e)}")
                    reply = msg.make_reply()
                    reply.set_metadata("phase", "scrape_error")
                    reply.body = json.dumps({
                        "error": str(e),
                        "query": query
                    })
                    await self.send(reply)

    async def setup(self):
        self.add_behaviour(self.CrawlBehaviour())
        logger.info(f"{self.jid} iniciado correctamente")
        

# Prueba directa del sistema
def test_simple_search():
    query = "¿De cuántos estados está compuesta la república?"
    urls = simple_search(query)
    print(f"URLs encontradas: {len(urls)}")
    for url in urls:
        print(f"- {url}")
    
    if urls:
        chunks = simple_scrape(urls[0], 2)
        print(f"\nChunks obtenidos: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:\n{chunk[:200]}...")
