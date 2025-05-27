class Coordinator:
    def __init__(self):
        self.redis = Redis()
    
    def handle_query(self, query: str):
        # 1. Enviar consulta al Agente de Búsqueda
        docs = self.redis.publish("retrieval", query)
        # 2. Enviar documentos al Agente de Generación
        answer = self.redis.publish("generation", (query, docs))
        return answer
    
# Agente Coordinador (orquesta el flujo)
class CoordinatorAgent(Agent):
    class RouteBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive()
            if msg.metadata["type"] == "historia":
                # Enviar a RetrievalAgent
                await self.send(Message(to="retrieval_agent@server", body=msg.body))
                
            elif msg.metadata["type"] == "matematicas":
                # Enviar a MathSolverAgent
                await self.send(Message(to="math_agent@server", body=msg.body))

# Configuración final
async def full_setup():
    agents = [
        CoordinatorAgent("coordinator@server", "pass"),
        RetrievalAgent("retrieval@server", "pass"),
        GenerationAgent("generation@server", "pass")
    ]
    for agent in agents:
        await agent.start()
    while True:
        await asyncio.sleep(1)