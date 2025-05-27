from prometheus_client import Counter, start_http_server
from spade.behaviour import CyclicBehaviour

REQUESTS_COUNTER = Counter('its_requests', 'Total consultas procesadas')

class SearchBehaviour(CyclicBehaviour):
    async def run(self):
        REQUESTS_COUNTER.inc()
        try:
            msg = await self.receive(timeout=30)
            # ... lógica existente ...
        except Exception as e:
            await self.send_error_report(e)

    async def send_error_report(self, error):
        msg = Message(to="admin_agent@your_xmpp_server.com")
        msg.body = f"Error en RetrievalAgent: {str(error)}"
        await self.send(msg)