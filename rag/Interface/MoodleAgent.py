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
        
        self.MOODLE_URL = "http://localhost"
        self.MOODLE_TOKEN = "56772b1ba299575696f3d809f25adbf9"
        self.USER_ID = 2
        self.ENDPOINT = f"http://localhost/webservice/rest/server.php"
        self.ROLE_STUDENT = 5
        self.last_messages={}

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

    def get_conversations(self):
        """Obtiene todos las conversaciones  del usuario"""
        try:
            conversations = self._make_request(
                'core_message_get_conversations',
                {'userid': self.USER_ID}
            )
            
            if 'exception' in conversations:
                return None
            
            return conversations
        except Exception as e:
            print(f"Error obteniendo mensajes: {e}")
            return None
    def get_messages(self,):
        """Obtiene los mensajes nuevos del usario"""
        conversations= self.get_conversations()
        all_messages = {}
        for conv in conversations.get("conversations", []):
        # Obtener el ID del otro participante (no el usuario actual)
            other_user_id = next(
                (member['id'] for member in conv['members'] 
                 if member['id'] != self.USER_ID),
                None
            )
        
            if not other_user_id:
                continue

            messages = self._make_request(
                'core_message_get_messages',
                {
                    'useridto': self.USER_ID,
                    'useridfrom': other_user_id,  # Usar USER ID real
                    'read': 0,
                    'limitfrom': self.USER_ID,           
                }
            )
            if messages and 'messages' in messages:
                    all_messages[conv['id']]=messages['messages']
        return all_messages
    
    def Mark_messages_read(self,message_id):
        return self._make_request(
            'core_message_mark_message_read',
            {
                'messageid': message_id
            }
        )

    def send_messages(self, message, conversation_id,timecreated):
        """Envía múltiples mensajes a una conversación"""
        params = {'conversationid': conversation_id}
        params['messages[0][text]'] = message
        params['messages[0][textformat]'] = 0
        self.last_messages[conversation_id]=timecreated+1
        return self._make_request(
            'core_message_send_messages_to_conversation',
            params
        )
    def Block_User(self,user_blocked):
        return self._make_request(
            'core_message_block_user',
            {
                'userid': self.USER_ID,
                'blockeduserid': user_blocked           
            }
        )
    def Unblock_User(self,user_unblocked):
       return self._make_request(
           'core_message_unblock_user',
           {
               'userid': self.USER_ID,
               'unblockeduserid': user_unblocked           
           }
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
            if not messages[id]: continue
            for message in messages[id]:
                
                self.agent.moodle_api.Block_User(message['useridfrom'])
                if message['text']:
                    msg = Message(
                        to="personality_analizer@localhost",
                        body=json.dumps({"query":message['text'],"user_id": message['useridfrom']}),
                        metadata={"phase":"analyzer"}
                    )
                    await self.send(msg)
                    print(f"[Monitor] Enviados {len(message)} mensajes al agente procesador")

                msg = await self.receive(timeout=10)
                if msg:
                    performative = msg.get_metadata("phase")
                    if performative == "final":
                        body = json.loads(msg.body)
                        final_answer = body.get("final_answer", "")
                        self.agent.moodle_api.send_messages(final_answer,id,message['timecreated'])
                        self.agent.moodle_api.Unblock_User(message['useridfrom'])
                        self.agent.moodle_api.Mark_messages_read(message['id'])
    #def send_User_Data_to_Profile(id):
        
