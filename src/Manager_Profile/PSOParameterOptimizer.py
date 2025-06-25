import numpy as np
import pyswarms as ps
class PSOParameterOptimizer:
    def __init__(self, profile, n_particles=15, iters=20):
        self.profile = profile
        self.n_particles = n_particles
        self.iters = iters
        
        # Espacio de búsqueda: [temperature, top_p, repetition_penalty]
        self.bounds = (
            np.array([0.1, 0.5, 1.0]),  # Límites inferiores
            np.array([1.0, 1.0, 2.0])   # Límites superiores
        )
    
    def optimize(self):
        # Configurar PSO
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles,
                                           dimensions=3,
                                           options=options,
                                           bounds=self.bounds)
        
        # Ejecutar optimización
        cost, pos = optimizer.optimize(self.fitness, iters=self.iters)
        engagement = self.profile.get('interaction_history',{}).get('avg_engagement', 0.5)
        prefs = self.profile['preferences']
        history = self.profile['interaction_history']
        history_prefs = self.profile['preferences']['history_specific']
        topic_prefs = self.profile['preferences']['topics']
        base_length = 150
        max_length=int(base_length + (engagement * 200))
        return {
            'temperature': float(pos[0]),
            'top_p': float(pos[1]),
            'repetition_penalty': float(pos[2]),
            'max_length':max_length,
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
    
    def fitness(self, params):
        # Simular rendimiento de parámetros
        engagement_scores = []
        
        for param_set in params:
            # Simular respuesta con estos parámetros
            # (En implementación real usar modelo predictivo)
            simulated_engagement = self.simulate_engagement(param_set)
            engagement_scores.append(1 - simulated_engagement)  # Minimizar 1-engagement
            
        return np.array(engagement_scores)
    
    def simulate_engagement(self, params):
        """Predice engagement basado en características de perfil"""
        # Modelo simplificado - en implementación real usar ML
        humor = self.profile['preferences']['communication']['humor']
        formality = self.profile['preferences']['communication']['formality']
        avg_engagement = self.profile['interaction_history']['engagement_metrics']['avg_engagement']
        
        # Parámetros: [temperature, top_p, repetition_penalty]
        temperature, top_p, rep_penalty = params
        
        # Modelo heurístico (reemplazar con modelo entrenado)
        engagement = (
            0.4 * (1 - abs(humor - temperature)) +
            0.3 * (1 - abs(formality - (1 - top_p))) +
            0.3 * avg_engagement
        )
        
        return np.clip(engagement, 0, 1)