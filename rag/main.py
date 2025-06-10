import asyncio
from utils.constants import (
    SEARCH_JID, EVAL_JID, PROMPT_JID, INITIATOR_JID, PASSWORDS, SERVER, SCRAPER_JID
)
from agents.query_initiator import QueryInitiatorAgent
from agents.searcher import SearchAgent
from agents.evaluator import EvaluationAgent
from agents.prompt_manager import PromptAgent
from agents.scrapper import ScraperAgent 


async def main():
    search_agent = SearchAgent(SEARCH_JID, PASSWORDS[SEARCH_JID])
    eval_agent = EvaluationAgent(EVAL_JID, PASSWORDS[EVAL_JID])
    prompt_agent = PromptAgent(PROMPT_JID, PASSWORDS[PROMPT_JID])
    scraper_agent = ScraperAgent(SCRAPER_JID, PASSWORDS[PROMPT_JID])
    initiator = QueryInitiatorAgent(INITIATOR_JID, PASSWORDS[INITIATOR_JID])
    
    await prompt_agent.start(auto_register=True)
    await eval_agent.start(auto_register=True)
    await search_agent.start(auto_register=True)
    await scraper_agent.start(auto_register=True)
    await asyncio.sleep(2)
    
    await initiator.start(auto_register=True)
    
    print("Sistema distribuido iniciado. Presiona Ctrl+C para detener.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Deteniendo agentes...")
        await initiator.stop()
        await search_agent.stop()
        await eval_agent.stop()
        await prompt_agent.stop()
        await scraper_agent.stop()

if __name__ == "__main__":
    asyncio.run(main())