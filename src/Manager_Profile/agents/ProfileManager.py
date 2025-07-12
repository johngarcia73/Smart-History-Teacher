from spade.agent import Agent
from spade.message import Message
from spade.behaviour import CyclicBehaviour
from spade.template import Template
from ..UserProfileManager import UserProfileManager
from ..UpdateProfile import InteractionBasedUpdater
from ..PSOParameterOptimizer import PSOParameterOptimizer
import numpy as np
import json
class profilemanageragent(Agent):
    
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.ProfileManager= UserProfileManager()
        self.UpdateManager= InteractionBasedUpdater(self.ProfileManager)

    async def setup(self):
        template = Template(metadata={"phase": "profile"})
        handleProfile= HandleProfileBehaviour()
        self.add_behaviour(handleProfile,template)
        handleUpdateProfile= HandleUpdateProfileBehaviour()
        template = Template(metadata={"phase": "interaction"})
        self.add_behaviour(handleUpdateProfile,template)

        pass
class HandleUpdateProfileBehaviour(CyclicBehaviour):
    async def run(self) :
        
        msg = await self.receive(timeout=10)
        
        if msg and msg.get_metadata("phase") == "interaction":
            data= json.loads(msg.body)
            User_Profile= self.agent.UpdateManager.update_profile(data["user_id"],data["interaction_data"])
            params=  await self.get_llm_parameters_op(User_Profile)
            send_msg= Message(
                to="prompt_agent@localhost",
                body=json.dumps(params),
                metadata= {"phase":"params"}
            )
            await self.send(send_msg)
    
    async def get_llm_parameters(self, profile):
        """Genera parámetros para el LLM basado en el perfil del usuario"""
        prefs = profile['preferences']
        history = profile['interaction_history']
        history_prefs = profile['preferences']['history_specific']
        topic_prefs = profile['preferences']['topics']
        # Parámetros principales para el LLM
        def _calculate_temperature(prefs, history):
            """
            Calcula el parámetro de temperatura para el LLM
            Rango: 0.0 (más determinista) a 1.0 (más creativo)
            """
            base_temp = 0.7
            # Ajustar por preferencia de humor
            humor_factor = prefs.get('humor', 0.5)
            temp_adjust = (humor_factor - 0.5) * 0.4
            # Ajustar por engagement histórico
            engagement = history.get('avg_engagement', 0.5)
            if engagement < 0.3:
                temp_adjust -= 0.1  # Más determinista para usuarios desconectados
            elif engagement > 0.7:
                temp_adjust += 0.1  # Más creativo para usuarios comprometidos
            return np.clip(base_temp + temp_adjust, 0.1, 1.0)
        def _calculate_top_p(prefs):
            """
            Calcula el parámetro top_p (nucleus sampling)
            Rango: 0.5 (más enfocado) a 1.0 (más diverso)
            """
            # Usuarios que prefieren formalidad quieren respuestas más enfocadas
            formality = prefs.get('formality', 0.5)
            return 0.9 - (formality * 0.4)
        def _calculate_repetition_penalty(history):
            """
            Calcula la penalización por repetición
            Rango: 1.0 (sin penalización) a 2.0 (máxima penalización)
            """
            # Usuarios con mucho engagement pueden tolerar más repetición
            engagement = history.get('avg_engagement', 0.5)
            return 1.2 + (0.6 * (1 - engagement))
        def _calculate_max_length(history):
            """
            Calcula la longitud máxima de respuesta
            Rango: 50-500 tokens
            """
            engagement = history.get('avg_engagement', 0.5)
            base_length = 150
            return int(base_length + (engagement * 200))
        
        
        params = {
            # Parametros Generales
            'temperature': _calculate_temperature(prefs, history),
            'top_p': _calculate_top_p(prefs),
            'repetition_penalty': _calculate_repetition_penalty(history),
            'max_length': _calculate_max_length(history),
            'style': prefs.get('communication_style', 'neutral'),
            'humor_level': prefs.get('humor', 0.5),
            'formality_level': prefs.get('formality', 0.5),
            #  Sitemas de temas
            'preferred_topics': list(prefs.get('topic_affinity', {}).keys()),
            'disliked_topics': prefs.get('disliked_topics', []),
            'response_types': history.get('preferred_response_types', ['standard']),
            'topic_affinity': topic_prefs['affinity'],
            # Parámetros específicos de historia
            'historiographical_approach': history_prefs['historiographical_approach'],
            'source_criticism': history_prefs['source_criticism'],
            'evidence_preference': history_prefs['evidence_preference'],
            'controversy_handling': history_prefs['controversy_handling'],
            'temporal_focus': history_prefs['temporal_focus'],
        }
        ## Añadir información de cluster si está disponible
        #if user_id in self.user_clusters:
        #    cluster_id = self.user_clusters[user_id]
        #    params['user_cluster'] = cluster_id
        #    params['cluster_description'] = self._generate_cluster_descriptions()[cluster_id]
        return params


    async def get_llm_parameters_op(self, profile):
        """Usa PSO para optimizar parámetros LLM"""

        if profile['metadata']['optimization_status'] == 'optimized':
            # Usar parámetros pre-optimizados
            return profile['metadata']['optimized_params']
    
        else:
            # Optimizar con PSO
            optimizer = PSOParameterOptimizer(profile)
            optimized_params = optimizer.optimize()
            
            # Actualizar perfil
            profile['metadata']['optimization_status'] = 'optimized'
            profile['metadata']['optimized_params'] = optimized_params
            self.agent.ProfileManager.save_profile(profile)
            
            return optimized_params
    
class HandleProfileBehaviour(CyclicBehaviour):
    async def run(self):
        
        msg= await self.receive(timeout=20)
        if msg and msg.metadata["phase"] == 'profile':
            data= json.loads(msg.body)
            profile= self.agent.ProfileManager.get_profile(data["user_id"])
            msg= Message(
                to="personality_analizer@localhost",
                body= json.dumps({
                    "profile":profile,
                    "user_id":data["user_id"],
                    "raw_query": data["raw_query"]}),
                metadata={"phase": "profile"}
            ) 
            await self.send(msg)           
               