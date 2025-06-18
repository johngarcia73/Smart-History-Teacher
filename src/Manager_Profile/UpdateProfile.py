import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import json
import os
from datetime import datetime

class InteractionBasedUpdater:
    def __init__(self, profile_manager):
        self.profile_manager = profile_manager
        self.user_clusters = {}
        self.cluster_centroids = {}
        self.cluster_last_updated = None
        
    def update_profile(self, user_id, interaction_data):
        """
        Actualiza el perfil del usuario basado en datos de interacción
        
        Args:
            user_id: Identificador del usuario
            interaction_data: Diccionario con datos de interacción
        """
        profile = self.profile_manager.get_profile(user_id)
        
        # Actualizar preferencias basadas en interacción
        self._update_communication_prefs(profile, interaction_data)
        self._update_topic_prefs(profile, interaction_data)
        self._update_engagement_metrics(profile, interaction_data)
        self._update_history_prefs(profile,interaction_data)

        # Guardar perfil actualizado
        self.profile_manager.save_profile(profile)
        
        # Actualizar clusters cada 24 horas o después de 100 interacciones
        if self._needs_cluster_update():
            self.update_user_clusters()
            
        return profile
    
    def _update_history_prefs(self, profile, interaction):
        """Actualiza preferencias específicas de historia"""
        prefs = profile['preferences']['history_specific']
        
        # Ajuste de enfoque historiográfico
        if 'historical_focus' in interaction:
            new_approach = interaction['historical_focus']
            prefs['historiographical_approach'] = new_approach
                
        # Ajuste de nivel de crítica
        if 'source_criticism_feedback' in interaction:
            level = prefs['source_criticism']
            levels = ["basic", "intermediate", "advanced"]
            current_idx = levels.index(level)
            
            if interaction['source_criticism_feedback'] == "too_simple" and current_idx < 2:
                prefs['source_criticism'] = levels[current_idx + 1]
            elif interaction['source_criticism_feedback'] == "too_complex" and current_idx > 0:
                prefs['source_criticism'] = levels[current_idx - 1]
    


    def _update_communication_prefs(self, profile, interaction):
        """Ajusta preferencias de comunicación basado en reacción"""
        prefs = profile['preferences']['communication']
        style_used = interaction.get('response_style', 'neutral')
        confidence= interaction.get('style_confidence',0.7)
        ADAPTATION_FACTOR=0.3
        DECAY_FACTOR=0.95
        # Ajustar humor basado en reacción a bromas
        if 'humor_score' in interaction:
            prefs['humor'] = (prefs['humor'] * (1 - ADAPTATION_FACTOR)) + (interaction['humor_score'] * ADAPTATION_FACTOR)
        
        # Ajustar formalidad basado en reacción
        if 'formality_level' in interaction:
            prefs['formality'] = (prefs['formality'] * (1 - ADAPTATION_FACTOR)) + (interaction['formality_level'] * ADAPTATION_FACTOR)
        

        update_style= {style: weight*DECAY_FACTOR for style, weight in prefs['style_weights'].items()}
        if style_used in update_style:
            update_style[style_used] +=confidence + (1-ADAPTATION_FACTOR)
        else:
            update_style[style_used] = confidence + (1-ADAPTATION_FACTOR)
        total = sum(update_style.values())           
        prefs['style_weights']= {style: weight / total for style, weight in update_style.items()}
        prefs['style_history'].append({
            'style':style_used,
            'timestamp':datetime.now().isoformat(),
            'confidence':confidence
        })
        prefs['style_history']=prefs['style_history'][-50:]
               
    def _update_topic_prefs(self, profile, interaction):
        """Actualiza preferencias de temas basado en interacción"""
        prefs = profile['preferences']['topics']
        topics = interaction.get('preferred_topics', [])
        if not topics:
            return
        
        # Calcular cambio en preferencia de temas
        decay_factor = 0.98  # Factor de decaimiento para temas no interactuados
        


        # Actualizar temas interactuados
        for topic in topics:
            topic_change = 0.1 if topics[topic] == 'positive' else -0.25
            if topic not in prefs['affinity']:
                prefs['affinity'][topic] = 0.5
            prefs['affinity'][topic] = np.clip(
                prefs['affinity'][topic] + topic_change, 0.0, 1.0
            )

        
        # Aplicar decaimiento a todos los temas
        for topic in prefs['affinity']:
            if topic not in topics:
                prefs['affinity'][topic] = (
                    (prefs['affinity'][topic] - 0.5) * decay_factor + 0.5
                )
    
    def _update_engagement_metrics(self, profile, interaction):
        """Actualiza métricas de engagement en el perfil"""
        history = profile['interaction_history']
        
        # Actualizar contadores
        history['total_interactions'] += 1
        history['last_interaction'] = datetime.now().isoformat()
        
        # Registrar tipo de respuesta preferida
        response_type = interaction.get('response_type', 'standard')
        if response_type not in history['preferred_response_types']:
            history['preferred_response_types'].append(response_type)
        
        # Calcular tiempo de compromiso normalizado
        engagement_time = min(interaction.get('engagement_time', 0), 300)  # Máximo 5 minutos
        normalized_engagement = engagement_time / 300  # Normalizado a [0,1]
        
        # Actualizar promedio móvil de engagement
        if 'avg_engagement' not in history:
            history['avg_engagement'] = normalized_engagement
        else:
            history['avg_engagement'] = 0.9 * history['avg_engagement'] + 0.1 * normalized_engagement
    
    def _needs_cluster_update(self):
        """Determina si se necesita actualizar los clusters de usuarios"""
        if not self.cluster_last_updated:
            return True
        hours_since_update = (datetime.now() - self.cluster_last_updated).total_seconds() / 3600
        return hours_since_update > 24
    
    def update_user_clusters(self, n_clusters=5):
        """Agrupa usuarios con preferencias similares usando K-means"""
        profiles = self._load_all_profiles()
        if len(profiles) < n_clusters * 2:  # Mínimo para clustering
            return
            
        # Preparar datos para clustering
        X = []
        user_ids = []
        feature_names = [
            'humor', 'formality', 'avg_engagement'
        ]
        
        for user_id, profile in profiles.items():
            prefs = profile['preferences']
            history = profile['interaction_history']
            
            features = [
                prefs.get('humor', 0.5),
                prefs.get('formality', 0.5),
                history.get('avg_engagement', 0.5)
            ]
            
            # Añadir afinidad por temas populares
            topic_features = self._get_topic_features(prefs.get('topic_affinity', {}))
            features.extend(topic_features)
            
            X.append(features)
            user_ids.append(user_id)
        
        # Ejecutar clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Almacenar resultados
        self.user_clusters = {}
        for user_id, cluster_id in zip(user_ids, clusters):
            self.user_clusters[user_id] = cluster_id
        
        self.cluster_centroids = kmeans.cluster_centers_
        self.cluster_last_updated = datetime.now()
        
        # Guardar metadatos de clusters
        self._save_cluster_metadata(feature_names + list(self._get_popular_topics().keys()))
    
    def _load_all_profiles(self):
        """Carga todos los perfiles desde el directorio"""
        profiles = {}
        base_dir = self.profile_manager.base_dir
        
        for filename in os.listdir(base_dir):
            if filename.endswith('.json'):
                user_id = filename[:-5]
                try:
                    profile = self.profile_manager.get_profile(user_id)
                    profiles[user_id] = profile
                except:
                    continue
                    
        return profiles
    
    def _get_topic_features(self, topic_affinity):
        """Obtiene características de temas para clustering"""
        popular_topics = self._get_popular_topics()
        features = []
        
        for topic in popular_topics:
            features.append(topic_affinity.get(topic, 0.5))
            
        return features
    
    def _get_popular_topics(self, min_users=5):
        """Identifica temas populares entre los usuarios"""
        topic_counts = defaultdict(int)
        profiles = self._load_all_profiles()
        
        for profile in profiles.values():
            topics = profile['preferences'].get('topic_affinity', {}).keys()
            for topic in topics:
                topic_counts[topic] += 1
                
        return {topic: count for topic, count in topic_counts.items() if count >= min_users}
    
    def _save_cluster_metadata(self, feature_names):
        """Guarda metadatos de clusters para referencia"""
        metadata = {
            'last_updated': self.cluster_last_updated.isoformat(),
            'feature_names': feature_names,
            'cluster_centroids': self.cluster_centroids.tolist(),
            'cluster_descriptions': self._generate_cluster_descriptions()
        }
        
        with open(os.path.join(self.profile_manager.base_dir, 'cluster_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_cluster_descriptions(self):
        """Genera descripciones automáticas de los clusters"""
        descriptions = []
        
        for i, centroid in enumerate(self.cluster_centroids):
            # Descripción basada en características principales
            desc_parts = []
            
            # Humor
            if centroid[0] > 0.7:
                desc_parts.append("prefieren humor")
            elif centroid[0] < 0.3:
                desc_parts.append("prefieren seriedad")
                
            # Formalidad
            if centroid[1] > 0.7:
                desc_parts.append("estilo formal")
            elif centroid[1] < 0.3:
                desc_parts.append("estilo casual")
                
            # Engagement
            if centroid[2] > 0.7:
                desc_parts.append("alto compromiso")
            elif centroid[2] < 0.3:
                desc_parts.append("bajo compromiso")
                
            descriptions.append(f"Cluster {i}: Usuarios que {', '.join(desc_parts)}")
            
        return descriptions
    
   