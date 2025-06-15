from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
    
# Agente Coordinador (orquesta el flujo)
class CoordinatorAgent(Agent):
    class RouteBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive()
            # Enviar a RetrievalAgent
            await self.send(Message(to="retrieval_agent@server", body=msg.body))
            #Enviar a GeneratorAgent
            msg= await self.receive()
            await self.send(Message(to="generation_agent@server", body=msg.body))

