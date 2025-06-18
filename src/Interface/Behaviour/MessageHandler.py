from spade.behaviour import CyclicBehaviour
import json
class MessageHandlerBehaviour(CyclicBehaviour):
    """Maneja solicitudes de otros agentes"""
    async def run(self):
        msg = await self.receive(timeout=10)
        if msg:
            performative = msg.get_metadata("phase")
            if performative == "final":
                await self.handle_request(msg)

    async def handle_request(self, msg):
        body = json.loads(msg.body)
        final_answer = body.get("final_answer", "")
        self.agent.moodle_api.send_messages(final_answer,body['conversation_id'])

        #action = body.get("action")
        #response = None
        #
        #try:
        #    if action == "send_messages":
        #        response = self.agent.moodle_api.send_messages(
        #            body['messages'], 
        #            body['conversation_id']
        #        )
        #    elif action == "create_course":
        #        response = self.agent.moodle_api.create_course(
        #            body['course_data']
        #        )
        #    elif action == "upload_file":
        #        response = self.agent.moodle_api.upload_file(
        #            body['file_data']
        #        )
        #except Exception as e:
        #    response = {"status": "error", "details": str(e)}
        #
        #reply = msg.make_reply()
        #reply.body = json.dumps(response or {"status": "unhandled_action"})
        #await self.send(reply)
