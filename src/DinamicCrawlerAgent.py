class DynamicCrawlerAgent:
    def __init__(self, knowledge_base: FAISS):
        self.knowledge_base = knowledge_base
    
    def check_and_update(self, query: str, threshold=0.8):
        # Si la confianza en la respuesta es baja, activa el crawler
        new_data = scrape_arxiv(query)
        self.knowledge_base.add(new_data)