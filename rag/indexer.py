# build_index.py
import os
import glob
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Asegurarse de tener descargado el paquete punkt de nltk (se descarga una sola vez)
#nltk.download('punkt')
#nltk.download('cess_esp')  # Corpus para español
#nltk.download('cess_esp_udep')  # Modelo POS para español
#nltk.download('spanish_grammars')  # Gramáticas para español

def load_documents_from_folder(folder_path):
    """
    Carga todos los archivos .txt de la carpeta indicada y devuelve una lista de documentos.
    """
    documents = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents

def chunk_text(text, max_sentences=2):
    """
    Divide un texto en chunks basados en oraciones.
    """
    sentences = sent_tokenize(text, language='spanish')
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks

def build_index(folder_path, index_file='faiss_index.bin', metadata_file='chunk_metadata.pickle'):
    """
    Procesa todos los archivos de data, crea los *chunks* y genera
    el índice FAISS junto con los metadatos (por ejemplo, qué chunk proviene de qué documento).
    """
    print(f"Cargando documentos desde: {folder_path}")
    documents = load_documents_from_folder(folder_path)
    chunks = []
    chunk_metadata = []
    2
    for doc_id, doc in enumerate(documents):
        doc_chunks = chunk_text(doc, max_sentences=2)
        for chunk in doc_chunks:
            chunks.append(chunk)
            chunk_metadata.append({'document_id': doc_id, 'text': chunk})
    
    for i, metadata in enumerate(chunk_metadata):
        metadata['source']  = 'local'
        
        
    print("Generando embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)
    
    print("Construyendo el índice FAISS...")
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    
    # Guardar el índice y los metadatos en disco
    print("Guardando el índice y los metadatos...")
    faiss.write_index(index, index_file)
    with open(metadata_file, 'wb') as f:
        pickle.dump(chunk_metadata, f)
    
    print("Se han generado y guardado el índice y los metadatos.")
  