import logging
import scrapy


logger = logging.getLogger("CrawlerAgentLogger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class DuckDuckGoSearchSpider(scrapy.Spider):
    name = "duckduckgo_search"
    custom_settings = {
         "DOWNLOAD_TIMEOUT": 10,
         "LOG_ENABLED": False
    }
    # Atributo de clase para almacenar los resultados
    results = []

    def __init__(self, query, limit=5, *args, **kwargs):
         super().__init__(*args, **kwargs)
         self.query = query
         self.limit = limit
         # Codificar la query para la URL de búsqueda
         query_encoded = query.replace(" ", "+")
         self.start_urls = [f"https://html.duckduckgo.com/html/?q={query_encoded}"]
         self.result_urls = []
         logger.debug(f"[DuckDuckGoSearchSpider] Iniciado para query: {query}")

    def parse(self, response):
         # Intentamos extraer enlaces usando selectores para la versión HTML de DuckDuckGo
         links = response.css("a.result__a::attr(href)").getall()
         if not links:
             links = response.css("div.result a::attr(href)").getall()
         
         for link in links:
             if len(self.result_urls) < self.limit:
                 self.result_urls.append(link)
             else:
                 break
         
         logger.debug(f"[DuckDuckGoSearchSpider] URLs encontradas: {self.result_urls}")
         # Almacenamos en el atributo de clase para poder recuperarlo luego
         DuckDuckGoSearchSpider.results = self.result_urls
         yield {"results": self.result_urls}
