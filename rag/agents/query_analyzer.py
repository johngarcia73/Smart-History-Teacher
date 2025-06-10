import spacy

class QueryAnalyzer:
    def __init__(self):
        try:
            print("Estoy en QueryAnalyzer")
            self.nlp = spacy.load("es_core_news_sm")
            print("Termine de instanciar QueryAnalyzer")
        except:
            raise ImportError("Modelo de spaCy para español no instalado. Ejecuta: python -m spacy download es_core_news_sm")
    
    def analyze(self, query):
        doc = self.nlp(query)
        
        # Detectar tipo de pregunta
        question_words = [token.text.lower() for token in doc]
        if "cuántos" in question_words or "cuántas" in question_words:
            return "factual"
        elif "cómo" in question_words and ("hacer" in question_words or "pasos" in question_words):
            return "procedural"
        elif "por qué" in query or "explica" in question_words:
            return "conceptual"
        
        # Detectar entidades
        entities = [ent.label_ for ent in doc.ents]
        if "PER" in entities or "DATE" in entities:
            return "factual"
        
        return "factual"  # Default