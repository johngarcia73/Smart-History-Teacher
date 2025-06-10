import json
import numpy as np

import os
import glob
import pickle
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Descargar recursos NLTK
#nltk.download('punkt', quiet=True)

def build_index(folder_path, index_file, metadata_file):
    """Construye el índice FAISS a partir de documentos"""
    logger.info(f"Construyendo índice desde: {folder_path}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logger.warning(f"Carpeta {folder_path} creada porque no existía.")
        return

    documents = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if content:
                documents.append(content)
            else:
                logger.warning(f"Documento vacío: {file_path}")

    if not documents:
        logger.warning("No se encontraron documentos válidos. Creando índice vacío.")
        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        metadata = []
        faiss.write_index(index, index_file)
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        return

    chunks = []
    metadata = []
    for doc_id, doc in enumerate(documents):
        sentences = sent_tokenize(doc, language='spanish')
        for i in range(0, len(sentences), 2):
            chunk = " ".join(sentences[i:i+2])
            chunks.append(chunk)
            metadata.append({
                'document_id': doc_id,
                'text': chunk,
                'source': 'local'
            })

    logger.info(f"Generando embeddings para {len(chunks)} chunks...")
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    batch_size = 100
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = embedder.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings).astype('float32')

    logger.info("Construyendo índice FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    logger.info(f"Guardando índice en {index_file}...")
    faiss.write_index(index, index_file)
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info("Índice construido exitosamente.")
    
    
def numpy_to_native(data):
    """Transforms NumPy types to Python native for JSON serialization"""
    if isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {k: numpy_to_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpy_to_native(item) for item in data]
    return data

def safe_json_dumps(data):
    """Serialize data"""
    return json.dumps(numpy_to_native(data))

