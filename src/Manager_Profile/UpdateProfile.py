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
        self.ga_optimizer = GeneticClusterOptimizer(pop_size=20, n_generations=50)
        
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
    

    
    def update_user_clusters(self):
        """Agrupa usuarios con preferencias similares usando K-means optimizado con GA"""
        profiles = self._load_all_profiles()
        X = []
        feature_names = [
            'humor', 'formality', 'avg_engagement'
        ]
        
        # Preparar datos para clustering
        user_ids = []
        for user_id, profile in profiles.items():
            prefs = profile['preferences']
            history = profile['interaction_history']
            
            features = [
                prefs['communication'].get('humor', 0.5),
                prefs['communication'].get('formality', 0.5),
                history.get('avg_engagement', 0.5)
            ]
            
            # Añadir afinidad por temas populares
            topic_features = self._get_topic_features(prefs['topics'].get('affinity', {}))
            features.extend(topic_features)
            
            X.append(features)
            user_ids.append(user_id)
        
        if not X:  # No hay datos para clusterizar
            return
            
        X = np.array(X)
        
        # Obtener nombres de características de temas
        popular_topics = self._get_popular_topics()
        feature_names.extend(list(popular_topics.keys()))
        self.feature_names = feature_names  # Guardar para descripciones
        
        # Optimizar parámetros de clustering con GA si hay suficientes datos
        if len(X) > 100:
            best_params = self.ga_optimizer.optimize(X)
            n_clusters = best_params['n_clusters']
            selected_features = best_params['features']
            X_selected = X[:, selected_features]
        else:
            n_clusters = 5
            selected_features = list(range(len(feature_names)))
            X_selected = X
        
        # Ejecutar clustering con parámetros optimizados
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_selected)
        
        # Almacenar resultados
        self.user_clusters = {}
        for user_id, cluster_id in zip(user_ids, clusters):
            self.user_clusters[user_id] = cluster_id
        
        self.cluster_centroids = kmeans.cluster_centers_
        self.selected_features = selected_features  # <-- Guardar características seleccionadas
        self.n_clusters = n_clusters  # <-- Guardar número de clusters
        self.cluster_last_updated = datetime.now()
        
        # Guardar metadatos de clusters
        self._save_cluster_metadata()

    # ... (otros métodos existentes sin cambios) ...



    def _needs_cluster_update(self):
        """Determina si se necesita actualizar los clusters de usuarios"""
        def contar_archivos(carpeta):
            # Lista solo archivos (no carpetas)
            archivos = [f for f in os.listdir(carpeta) 
               if os.path.isfile(os.path.join(carpeta, f))]
            return len(archivos)
        ruta="./src/Manager_Profile/user_profiles"
        total_perfil= contar_archivos(ruta) > 50
        hours_since_update=0
        if not self.cluster_last_updated and total_perfil:
            return True
        if self.cluster_last_updated:
            hours_since_update = (datetime.now() - self.cluster_last_updated).total_seconds() / 3600
        
        return hours_since_update > 24 and total_perfil
    
    
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
    
    def _save_cluster_metadata(self):
        """Guarda metadatos de clusters para referencia"""
        metadata = {
            'last_updated': self.cluster_last_updated.isoformat(),
            'n_clusters': self.n_clusters,  # <-- Nuevo
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,  # <-- Nuevo
            'cluster_centroids': self.cluster_centroids.tolist(),
            'cluster_descriptions': self._generate_cluster_descriptions()
        }
        
        with open(os.path.join(self.profile_manager.base_dir, 'cluster_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_cluster_descriptions(self):
        """Genera descripciones automáticas de los clusters usando características seleccionadas"""
        descriptions = []
        
        for i, centroid in enumerate(self.cluster_centroids):
            desc_parts = []
            centroid_full = np.zeros(len(self.feature_names))
            
            # Mapear características seleccionadas al espacio completo
            for j, feature_idx in enumerate(self.selected_features):
                centroid_full[feature_idx] = centroid[j]
            
            # Interpretar características basado en valores
            if 0 in self.selected_features and centroid_full[0] > 0.7:
                desc_parts.append("prefieren humor")
            elif 0 in self.selected_features and centroid_full[0] < 0.3:
                desc_parts.append("prefieren seriedad")
                
            if 1 in self.selected_features and centroid_full[1] > 0.7:
                desc_parts.append("estilo formal")
            elif 1 in self.selected_features and centroid_full[1] < 0.3:
                desc_parts.append("estilo casual")
                
            if 2 in self.selected_features and centroid_full[2] > 0.7:
                desc_parts.append("alto compromiso")
            elif 2 in self.selected_features and centroid_full[2] < 0.3:
                desc_parts.append("bajo compromiso")
            
            # Identificar temas más relevantes
            topic_indices = self.selected_features.copy()
            # Eliminar índices de características base (humor, formalidad, engagement)
            for idx in [0, 1, 2]:
                if idx in topic_indices:
                    topic_indices.remove(idx)
            
            # Ordenar temas por importancia en el centroide
            if topic_indices:
                topic_importances = [(self.feature_names[idx], centroid_full[idx]) 
                                    for idx in topic_indices]
                topic_importances.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
                
                # Tomar hasta 3 temas más relevantes
                for topic, importance in topic_importances[:3]:
                    if importance > 0.6:
                        desc_parts.append(f"afinidad por {topic}")
                    elif importance < 0.4:
                        desc_parts.append(f"baja afinidad por {topic}")
            
            descriptions.append(f"Cluster {i}: Usuarios que {', '.join(desc_parts)}")
            
        return descriptions

from deap import base, creator, tools, algorithms
import numpy as np
class GeneticClusterOptimizer:
    def __init__(self, pop_size=20, n_generations=50):
        self.pop_size = pop_size
        self.n_generations = n_generations
    
    def optimize(self, data):
        # Configurar algoritmo genético
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        n_features = data.shape[1]
        
        # Configuración de genes
        toolbox.register("attr_bool", np.random.randint, 0, 2)
        toolbox.register("attr_k", np.random.randint, 2, 10)  # k entre 2-10
        
        # Crear individuo [k, feature1, feature2, ..., featureN]
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_k, 
                          lambda: [toolbox.attr_bool() for _ in range(n_features)]),
                         1)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate, data=data)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Ejecutar algoritmo
        pop = toolbox.population(n=self.pop_size)
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                            ngen=self.n_generations, verbose=False)
        
        # Obtener mejor solución
        best_ind = tools.selBest(pop, 1)[0]
        return {
            'n_clusters': best_ind[0],
            'features': [i for i, val in enumerate(best_ind[1:]) if val == 1]
        }
    
    def evaluate(self, individual, data):
        k = individual[0]
        selected_features = [i for i, val in enumerate(individual[1:]) if val == 1]
        
        if len(selected_features) < 2 or k < 2:
            return -1000,  # Penalizar soluciones inválidas
            
        X = data[:, selected_features]
        
        # Calcular métrica de clustering (ej: silhouette score)
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        try:
            score = silhouette_score(X, labels)
            return score,
        except:
            return -1,

    def mutate(self, individual, indpb):
        # Mutación para k
        if np.random.rand() < indpb:
            individual[0] = np.clip(individual[0] + np.random.choice([-1, 1]), 2, 10)
        
        # Mutación para características
        for i in range(1, len(individual)):
            if np.random.rand() < indpb:
                individual[i] = 1 - individual[i]  # Flip binario
                
        return individual, 