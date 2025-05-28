# Paso 0: Importar librerías necesarias
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import requests
import json
import os
from dotenv import load_dotenv

# Lista de documentos (ajústalos según tus necesidades)
documents = [
    "Texto del documento 1 sobre álgebra y ecuaciones. Una ecuación cuadrática se resuelve con el uso de una fórmula específica.",
    "Contenido del documento 2 sobre geometría y cálculos espaciales. Se discute el teorema de Pitágoras y se enseña a calcular áreas y volúmenes de diversas figuras geométricas. Además, se abordan conceptos de geometría analítica.",
    "Descripción del documento 3 sobre cálculo diferencial e integral. En este documento se abordan conceptos de derivadas y reglas de integración, junto con ejemplos detallados de aplicación en problemas reales."
]

# Función para dividir el texto en "chunks" (fragmentos) basados en oraciones
def chunk_text(text, max_sentences=2):
    sentences = sent_tokenize(text, language='spanish')
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        chunks.append(chunk)
    return chunks

# Crear chunks para cada documento y guardar metadatos
chunks = []
chunk_metadata = []  # Para identificar a qué documento pertenece cada chunk, etc.
for doc_id, doc in enumerate(documents):
    doc_chunks = chunk_text(doc, max_sentences=2)
    for chunk in doc_chunks:
        chunks.append(chunk)
        chunk_metadata.append({'document_id': doc_id, 'text': chunk})

# Generar embeddings para cada chunk usando SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)

# Construir índice FAISS basado en la distancia Euclidiana (L2)
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# Función para recuperar los chunks más relevantes según la consulta
def retrieve_chunks(query, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = []
    for idx in indices[0]:
        meta = chunk_metadata[idx]
        retrieved_chunks.append(meta['text'])
    return retrieved_chunks

# Función de generación utilizando la API de inferencia online de Hugging Face
def generate_answer(query, retrieved_chunks):
    # Combinar los chunks recuperados en un único contexto
    context = "\n".join(retrieved_chunks)
    prompt = (
        f"Pregunta: {query}\n"
        f"Contexto: {context}\n"
        f"Respuesta:"
    )
    
    load_dotenv()
    hf_api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    hf_token = os.getenv("HF_API_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "num_beams": 5,
            "early_stopping": True
        }
    }
    
    try:
        response = requests.post(hf_api_url, headers=headers, json=payload)
        if response.status_code != 200:
            return f"Error al consultar el modelo: {response.status_code} - {response.text}"
            
        result = response.json()
        # Se espera que la respuesta sea una lista de diccionarios. Se verifican dos posibles claves.
        if isinstance(result, list):
            if "generated_text" in result[0]:
                return result[0]["generated_text"].strip()
            elif "summary_text" in result[0]:
                return result[0]["summary_text"].strip()
            else:
                return f"Respuesta inesperada: {result}"
        else:
            return f"Respuesta inesperada: {result}"
    except requests.exceptions.RequestException as e:
        return f"Error en la consulta al modelo: {str(e)}"

# Ejemplo de uso:
query = "¿Cómo se resuelven las ecuaciones cuadráticas?"
retrieved = retrieve_chunks(query, top_k=2)
answer = generate_answer(query, retrieved)

print(answer)
