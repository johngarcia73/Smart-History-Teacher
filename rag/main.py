import asyncio
from utils.constants import (
    SEARCH_JID, EVAL_JID, PROMPT_JID, INITIATOR_JID, PASSWORDS, SERVER, PERSONALITY_JID, CRAWLER_JID,MOODLE_JID,PROFILE_JID,LOG_DIR
)
from agents.query_initiator import QueryInitiatorAgent
from agents.searcher import SearchAgent
from agents.evaluator import EvaluationAgent
from agents.prompt_manager import PromptAgent
#from agents.scrapper import ScraperAgent 
from agents.crawler import CrawlerAgent
from Interface.MoodleAgent import MoodleAgent
from src.agents.ProfileManager import profilemanageragent
import os
import logging
from logging.handlers import RotatingFileHandler
from utils.logging import configure_logging
from ontology.ontology import OntologyManager
from src.agents.Personality_Analizer import PersonalityAnalyzerAgent


print("Ontología histórica construida exitosamente!")

async def main():
    
    current_log = configure_logging()
    logger = logging.getLogger(__name__)
        
    search_agent = SearchAgent(SEARCH_JID, PASSWORDS[SEARCH_JID])
    eval_agent = EvaluationAgent(EVAL_JID, PASSWORDS[EVAL_JID])
    prompt_agent = PromptAgent(PROMPT_JID, PASSWORDS[PROMPT_JID])
    #scraper_agent = ScraperAgent(SCRAPER_JID, PASSWORDS[SCRAPER_JID])
    crawler_agent = CrawlerAgent(CRAWLER_JID, PASSWORDS[CRAWLER_JID])
    #initiator = QueryInitiatorAgent(INITIATOR_JID, PASSWORDS[INITIATOR_JID])
    Moodle_Agent= MoodleAgent(MOODLE_JID,PASSWORDS[MOODLE_JID])
    profile_Agent=profilemanageragent(PROFILE_JID,PASSWORDS[PROFILE_JID])
    Personality_Analyzer_agent= PersonalityAnalyzerAgent(PERSONALITY_JID,PASSWORDS[PERSONALITY_JID])

    await Personality_Analyzer_agent.start(auto_register=True)
    await prompt_agent.start(auto_register=True)
    await search_agent.start(auto_register=True)
    await eval_agent.start(auto_register=True)
    await crawler_agent.start(auto_register=True)
    await profile_Agent.start(auto_register=True)
    #await scraper_agent.start(auto_register=True)
    await asyncio.sleep(2)
    
    await Moodle_Agent.start(auto_register=True)
    #await initiator.start(auto_register=True)
    
    print("Sistema distribuido iniciado. Presiona Ctrl+C para detener.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Deteniendo agentes...")
        #await initiator.stop()
        await Moodle_Agent.stop()
        await Personality_Analyzer_agent.stop()
        await search_agent.stop()
        await eval_agent.stop()
        await prompt_agent.stop()
        await crawler_agent.stop()
        await profile_Agent.stop()
        #await scraper_agent.stop()

if __name__ == "__main__":
    asyncio.run(main())