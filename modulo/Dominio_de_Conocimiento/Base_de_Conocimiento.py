from transformers import AutoTokenizer, AutoModel
import re
import torch
import numpy as np
import faiss


def roman_to_int(roman: str) -> int:
    roman_numerals = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    total = 0
    prev_value = 0
    for char in reversed(roman.upper()):
        value = roman_numerals.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    return total


# 1. Cargar un modelo y tokenizador adecuado para historia (multilingüe o entrenado en corpus histórico)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")  # Bueno para términos en múltiples idiomas
# Alternativas especializadas:
# - "dbmdz/bert-historic-multilingual-cased" (entrenado en textos históricos)
# - "EleutherAI/historic-bert" (si está disponible)
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

# 2. Preprocesamiento específico para historia
    # Normalizar fechas (ej: "siglo XVI" → "siglo 16")
def preprocess_historical_text(text):
    text = re.sub(r"siglo\s+([IVXLCDM]+)", lambda m: f"siglo {roman_to_int(m.group(1))}", text)
    
    # Unificar términos comunes (ej: "Segunda Guerra Mundial" → "WW2" si es necesario)
    text = text.replace("Segunda Guerra Mundial", "WW2")
    
    # Eliminar corchetes/citas de documentos antiguos (ej: [Nota del traductor])
    text = re.sub(r"\[.*?\]", "", text)
    
    return text

# 3. Tokenización con manejo de entidades históricas
def tokenize_historical_doc(text):
    text = preprocess_historical_text(text)
    tokens = tokenizer.tokenize(text)
    return tokens


# Cargar modelo y tokenizador

def generate_embeddings(text: str) -> np.ndarray:
    # Tokenización y preprocesamiento
    inputs = tokenize_historical_doc(text)

    # Generar embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtener el embedding del documento (promedio de los tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Ejemplo
documento = "La Revolución Francesa comenzó en 1789."
embedding = generate_embeddings(documento)
print("Embedding shape:", embedding.shape)  # Ej: (768,)



def Corpus_Vectorial(text):

    # Crear índice FAISS (dimensiones deben coincidir con tu embedding)
    dimension = 768  # BERT-base tiene 768 dimensiones
    index = faiss.IndexFlatL2(dimension)  # Índice para búsqueda por similitud L2

    # Supongamos que tienes una lista de embeddings de tus documentos
    corpus_embeddings = generate_embeddings(text)  # Lista de arrays numpy
    index.add(np.array(corpus_embeddings))  # Añadir todos los embeddings al índice

    # Guardar el índice en disco
    faiss.write_index(index, "corpus_historico.faiss")
    # Antes de añadirlos al índice FAISS
    faiss.normalize_L2(corpus_embeddings)
    index.add(corpus_embeddings)