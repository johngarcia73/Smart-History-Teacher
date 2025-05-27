import GenerationAgent
import RetrievalAgent

async def setup_agents():
    # Configurar Agente de Búsqueda
    retrieval = RetrievalAgent(
        "retrieval_agent@your_xmpp_server.com", 
        "password123"
    )
    retrieval.behaviours.append(RetrievalAgent.SearchBehaviour())
    
    # Configurar Agente de Generación
    generation = GenerationAgent(
        "generation_agent@your_xmpp_server.com",
        "password123"
    )
    generation.behaviours.append(GenerationAgent.GenerateBehaviour())
    
    # Iniciar agentes
    await retrieval.start()
    await generation.start()

# Ejecutar en loop asyncio
import asyncio
asyncio.run(setup_agents())

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