import asyncio
import json
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
from utils.constants import SEARCH_JID, EVAL_JID, PROMPT_JID, PASSWORDS
from utils.testingquestions import testingqueries
#from agents.evaluator import max_list
import numpy as np
import logging
class QueryInitiatorAgent(Agent):
    class QueryBehaviour(OneShotBehaviour):
        async def run(self):
            queries = [
                #"¿De cuántos estados está compuesta la república?",
                #"Logros militares de Napoleón Bonaparte"
                #"El loco más loco de la ciudad maldita de Notre Dame"
                #"Chocolate Rastamemba, el Fari, la bella y la bestia"
                #"Cómo fue la Revolución Francesa?"
                #"Galileo, a principios del siglo XVII, utilizara un telescopio"
                #"Isaac Newton, el último mago de Chile, hizo su aparición en el circo de Inglaterra"
                #"Revolución Darwiniana"
                "WMOHs s jwswiba DJWQI(3829 ) #NI@"
            ]
            
            logger = logging.getLogger(__name__)
            
            for query in queries:
                msg = Message(to=SEARCH_JID)
                msg.set_metadata("phase", "query")
                msg.body = json.dumps({"query": query})
                await self.send(msg)
                
                # Esperar respuesta
                final_msg = await self.receive(timeout=40)
                if final_msg and final_msg.get_metadata("phase") == "final":
                    try:
                        data = json.loads(final_msg.body)
                        final_answer = data.get("final_answer", "")
                        
                        if "No se pudo generar" in final_answer:
                            print(f"QueryInitiator: Error en la respuesta para '{query}'")
                        else:
                            print("\n------------------------------")
                            print(f"QueryInitiator: Respuesta para '{query}':")
                            print(final_answer)
                            print("------------------------------\n")
                    
                    except json.JSONDecodeError:
                        print(f"QueryInitiator: Respuesta inválida para '{query}'")
                else:
                    print(f"QueryInitiator: No se recibió respuesta para '{query}'")
                
                await asyncio.sleep(3)
                
            #logger.info(f"La media de score es con sliding window {np.mean(max_list)}")
            await self.agent.stop()

    async def setup(self):
        self.add_behaviour(self.QueryBehaviour())
        print(f"{self.jid} iniciado correctamente")