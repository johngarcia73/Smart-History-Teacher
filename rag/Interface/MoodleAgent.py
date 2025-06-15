import requests
import os

from spade.behaviour import PeriodicBehaviour
#from ..utils.constants import SEARCH_JID
from spade.agent import Agent
from spade.template import Template
from spade.message import Message

import json
import time
import asyncio

class MoodleAgent(Agent):
    def __init__(self,user,password):
        super().__init__(user,password)
        self.moodle_api = MoodleAPI()

    async def setup(self):
        print(f"Agente Moodle iniciado: {self.jid}")
        
        # Comportamiento para monitoreo de mensajes
        monitor = MoodleMonitorBehaviour(period=30)  # Revisa cada 30 segundos
        self.add_behaviour(monitor)
        
        # Comportamiento para manejar solicitudes
        #template = Template()
        #template.set_metadata("performative", "request")
        #self.add_behaviour(MessageHandlerBehaviour(), template)

class MoodleAPI:
    """Clase encapsulada para interacciones con la API de Moodle"""
    def __init__(self):
        self.MOODLE_URL = os.getenv("url").rstrip('/')
        self.MOODLE_TOKEN = os.getenv("Moodle_token")
        self.USER_ID = os.getenv("Moodle_ITS_ID")
        self.ENDPOINT = f"{os.getenv("url")}/webservice/rest/server.php"
        self.ROLE_STUDENT = 5

    def _make_request(self, wsfunction, params):
        base_params = {
            "wstoken": self.MOODLE_TOKEN,
            "wsfunction": wsfunction,
            "moodlewsrestformat": "json"
        }
        payload = {**base_params, **params}
        
        try:
            response = requests.post(self.ENDPOINT, data=payload, timeout=10)
            print(response)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error en la petición a Moodle: {e}")
            return None

    def get_messages(self):
        """Obtiene todos los mensajes nuevos del usuario"""
        try:
            conversations = self._make_request(
                'core_message_get_conversations',
                {'userid': self.USER_ID}
            )
            
            if 'exception' in conversations:
                return None

            all_messages = {}
            for conv in conversations.get("conversations", []):
                messages = self._make_request(
                    'core_message_get_conversation_messages',
                    {'currentuserid': self.USER_ID, 'convid': conv['id']}
                )
                if messages and 'messages' in messages:
                    all_messages[conv['id']]=messages['messages']
            return all_messages
        except Exception as e:
            print(f"Error obteniendo mensajes: {e}")
            return None

    def send_messages(self, message, conversation_id):
        """Envía múltiples mensajes a una conversación"""
        params = {'conversationid': conversation_id}
        params['messages[0][text]'] = message
        params['messages[0][textformat]'] = 0
        return self._make_request(
            'core_message_send_messages_to_conversation',
            params
        )

    def create_course(self, course_data):
        """Crea un nuevo curso"""
        return self._make_request(
            'core_course_create_courses',
            {'courses[0]': course_data}
        )

    def upload_file(self, file_data):
        """Sube un archivo al repositorio de Moodle"""
        return self._make_request(
            'core_files_upload',
            file_data
        )


class MoodleMonitorBehaviour(PeriodicBehaviour):
    """Comportamiento para monitorear mensajes periódicamente"""
    async def run(self):
        messages = self.agent.moodle_api.get_messages()
        for id in messages:
            #msg= self.send_User_Data_to_Profile(messages[id])
            msg= Message(to="profileManageragent@localhost",body= f"{id}",metadata={"phase":"profile"})
            await self.send(msg)
            for message in messages[id]:
                if message['text']:
                    msg = Message(
                        to="search_agent@localhost",
                        body=json.dumps({"query":message['text']}),
                        metadata={"phase":"query"}
                    )
                    await self.send(msg)
                    print(f"[Monitor] Enviados {len(message)} mensajes al agente procesador")

                msg = await self.receive(timeout=10)
                if msg:
                    performative = msg.get_metadata("phase")
                    if performative == "final":
                        body = json.loads(msg.body)
                        final_answer = body.get("final_answer", "")
                        self.agent.moodle_api.send_messages(final_answer,id)
    #def send_User_Data_to_Profile(id):
        
