from spade.agent import Agent
from spade.behaviour import PeriodicBehaviour
from scrapy.crawler import CrawlerProcess
import DinamicCrawlerAgent

class CrawlerAgent(Agent):
    class UpdateCheckBehaviour(PeriodicBehaviour):
        async def run(self):
            # Lógica de detección de brechas
            if self.agent.knowledge_gap_detected():
                await self.agent.start_crawler()

    async def setup(self):
        self.add_behaviour(self.UpdateCheckBehaviour(period=3600))  # Cada hora

    def knowledge_gap_detected(self):
        # Implementar lógica con métricas de confianza
        return confidence < 0.8

    async def start_crawler(self):
        process = CrawlerProcess(settings={
            'ITEM_PIPELINES': {'__main__.FaissPipeline': 300},
            'USER_AGENT': 'Mozilla/5.0 (ITS Historico)'
        })
        process.crawl(DinamicCrawlerAgent)
        process.start()
