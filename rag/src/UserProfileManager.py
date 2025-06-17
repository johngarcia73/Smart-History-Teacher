import os
import json
import uuid
from datetime import datetime

class UserProfileManager:
    def __init__(self):
        self.base_dir = "./src/user_profiles"
        os.makedirs(self.base_dir, exist_ok=True)
    
    def _get_profile_path(self, user_id):
        """Genera la ruta del archivo de perfil"""
        return os.path.join(self.base_dir, f"{user_id}.json")
    
    def _create_default_profile(self, user_id):
        """Crea un perfil con valores predeterminados"""
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "preferences": {
                # Parámetros generales
                "communication": {
                    "humor": 0.5,
                    "formality": 0.5,
                    "style": "neutral",
                    "tone": "neutral"
                },
                # Parámetros específicos de historia
                "history_specific": {
                    "historiographical_approach": "social",  # Enfoque historiográfico
                    "source_criticism": "intermediate",      # Nivel de crítica: basic/intermediate/advanced
                    "evidence_preference": ["primary", "visual"],  # Tipos de evidencia preferida
                    "controversy_handling": "neutral",       # neutral/critical/debate
                    "temporal_focus": "thematic"             # chronological/thematic/comparative
                },
                # Sistema de temas
                "topics": {
                    "preferred": [],
                    "disliked": [],
                    "affinity": {}  # {tema: valor 0-1}
                }
            },
            "interaction_history": {
                "total_interactions": 0,
                "last_interaction": None,
                "preferred_response_types": [],
                "engagement_metrics": {
                    "avg_engagement": 0.5,
                    "last_engagement": 0.5
                }
            },
            "metadata": {
                "is_new_user": True,
                "profile_version": "2.0",
                "optimization_status": "unoptimized"  # Para metaheurísticas futuras
            }
        }
    
    def user_profile_exists(self, user_id):
        """Verifica si un perfil existe"""
        return os.path.exists(self._get_profile_path(user_id))
    
    def create_profile(self, user_id=None, initial_preferences=None):
        """Crea un nuevo perfil de usuario"""
        user_id = user_id or str(uuid.uuid4())
        
        if self.user_profile_exists(user_id):
            raise ValueError(f"El perfil para {user_id} ya existe")
        
        profile = self._create_default_profile(user_id)
        
        # Aplica preferencias iniciales si existen
        if initial_preferences:
            for key, value in initial_preferences.items():
                if key in profile["preferences"]:
                    profile["preferences"][key] = value
        
        self.save_profile(profile)
        return profile
    
    def get_profile(self, user_id):
        """Obtiene el perfil de un usuario"""
        if not self.user_profile_exists(user_id):
            return self.create_profile(user_id)
        
        with open(self._get_profile_path(user_id), 'r') as f:
            return json.load(f)
    
    def save_profile(self, profile):
        """Guarda el perfil en disco"""
        profile["last_updated"] = datetime.now().isoformat()
        with open(self._get_profile_path(profile["user_id"]), 'w') as f:
            json.dump(profile, f, indent=2)
    
    def update_profile(self, user_id, updates):
        """Actualiza un perfil existente"""
        profile = self.get_profile(user_id)
        
        # Actualización recursiva de campos
        def update_recursive(target, update_data):
            for key, value in update_data.items():
                if isinstance(value, dict) and key in target:
                    update_recursive(target[key], value)
                else:
                    target[key] = value
        
        update_recursive(profile, updates)
        profile["metadata"]["is_new_user"] = False
        self.save_profile(profile)
        return profile


# --- Ejemplo de uso ---
if __name__ == "__main__":
    profile_manager = UserProfileManager()
    
    # Caso 1: Crear nuevo perfil
    new_user_id = "user_12345"
    print("\nCaso 1: Creando nuevo perfil...")
    new_profile = profile_manager.create_profile(
        user_id=new_user_id,
        initial_preferences={
            "humor": 0.8,
            "communication_style": "gossip_jokes",
            "preferred_topics": ["tecnología", "videojuegos"]
        }
    )
    print(json.dumps(new_profile, indent=2))
    
    # Caso 2: Obtener perfil existente
    print("\nCaso 2: Obteniendo perfil existente...")
    existing_profile = profile_manager.get_profile(new_user_id)
    print(f"Estilo de comunicación: {existing_profile['preferences']['communication_style']}")
    
    # Caso 3: Actualizar perfil
    print("\nCaso 3: Actualizando perfil...")
    updated_profile = profile_manager.update_profile(
        new_user_id,
        {
            "preferences": {
                "formality": 0.3,
                "preferred_topics": ["IA", "robótica"]
            },
            "interaction_history": {
                "total_interactions": 5,
                "preferred_response_types": ["humor", "analogías"]
            }
        }
    )
    print(json.dumps(updated_profile, indent=2))
    
    # Caso 4: Get or create
    print("\nCaso 4: Get or create para nuevo usuario...")
    auto_user = profile_manager.get_or_create_profile("auto_generated_user")
    print(f"ID de usuario generado: {auto_user['user_id']}")