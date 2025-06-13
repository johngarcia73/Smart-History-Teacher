import os
import glob
import pickle
import faiss
import requests
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import time
import logging
import json
import numpy as np
import random
from collections import defaultdict

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración: carpeta donde se guardarán los libros descargados
BOOKS_FOLDER = "data"
os.makedirs(BOOKS_FOLDER, exist_ok=True)

# Umbral mínimo de libros que se desean tener descargados para indexar
MIN_BOOKS = 2

def download_books_from_internet_archive(max_books=100):
    """Descarga libros de historia en español de Internet Archive usando la API avanzada y el endpoint de metadata.
       Se descargan aquellos libros que tengan más descargas (ordenados de mayor a menor) y se evitan los ya existentes.
    """
    logger.info(f"Iniciando descarga de {max_books} libros de historia en español desde Internet Archive...")
    
    downloaded_files = []
    
    search_url = "https://archive.org/advancedsearch.php"
    params = {
        'q': 'subject:(historia) AND mediatype:(texts) AND language:(spa)',
        'fl[]': ['identifier', 'title', 'downloads'],
        'sort[]': 'downloads desc',
        'rows': str(max_books),
        'page': '1',
        'output': 'json'
    }
    
    try:
        logger.info(f"Realizando búsqueda: {params['q']}")
        response = requests.get(search_url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        data = response.json()
        
        if 'response' not in data or 'docs' not in data['response']:
            logger.error("Respuesta inesperada de la API de búsqueda")
            return downloaded_files
        
        docs = data['response']['docs']
        logger.info(f"Encontrados {len(docs)} resultados")
        
        for doc in docs:
            identifier = doc.get('identifier')
            title = doc.get('title', identifier)
            downloads = int(doc.get('downloads', 0))
            logger.info(f"Procesando: {title} (Descargas: {downloads})")
            
            # Definir la ruta destino del archivo
            file_path = os.path.join(BOOKS_FOLDER, f"{identifier}.txt")
            if os.path.exists(file_path):
                logger.info(f"El documento {identifier} ya está descargado. Se omite.")
                downloaded_files.append(file_path)
                continue
            
            meta_url = f"https://archive.org/metadata/{identifier}"
            meta_response = requests.get(meta_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            meta_response.raise_for_status()
            meta_data = meta_response.json()
            
            if 'files' not in meta_data:
                logger.warning(f"No se encontraron archivos para {identifier}")
                continue
            
            txt_files = [f for f in meta_data['files'] 
                         if f.get('name', '').lower().endswith('.txt') and int(f.get('size', '0')) > 10240]
            if not txt_files:
                logger.warning(f"No se encontraron archivos .txt válidos en {identifier}")
                continue
            
            largest_file = max(txt_files, key=lambda x: int(x['size']))
            file_name = largest_file['name']
            download_url = f"https://archive.org/download/{identifier}/{file_name}"
            
            logger.info(f"Descargando {file_name} ({int(largest_file['size'])/1024:.1f} KB) de {identifier}...")
            try:
                with requests.get(download_url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
                    r.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as download_err:
                logger.error(f"Error al descargar {identifier}: {download_err}")
                continue
            
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                text = raw_data.decode('utf-8', errors='replace')
            except Exception as decode_err:
                logger.error(f"Error decodificando {file_path}: {decode_err}")
                continue
            
            if not text.strip() or len(text) < 200:
                logger.warning(f"Archivo {identifier} parece vacío o muy corto")
                continue
            
            downloaded_files.append(file_path)
            logger.info(f"Descarga completada: {file_path}")
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error en la descarga: {str(e)}", exc_info=True)
    
    return downloaded_files

def load_documents_from_folder(folder_path):
    """Carga todos los archivos .txt de la carpeta y devuelve una lista de documentos."""
    documents = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
                if content.strip():
                    documents.append(content)
                else:
                    logger.warning(f"Archivo vacío: {file_path}")
        except Exception as e:
            logger.error(f"Error leyendo {file_path}: {str(e)}")
    return documents

def chunk_text(text, max_sentences=2):
    """Divide un texto en chunks basados en oraciones (usando el tokenizer en español)."""
    try:
        sentences = sent_tokenize(text, language='spanish')
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i + max_sentences])
            chunks.append(chunk)
        return chunks
    except Exception as e:
        logger.error(f"Error en chunk_text: {str(e)}")
        return []

def build_index(folder_path, index_file='faiss_index.bin', metadata_file='chunk_metadata.pickle'):
    """
    Procesa todos los archivos .txt de la carpeta, crea los chunks y genera el índice FAISS.
    Si el índice ya existe, no hace nada.
    Si la carpeta tiene menos de un número determinado de libros, descarga hasta llegar a ese número.
    Durante el entrenamiento del índice se usa un subconjunto de embeddings, extrayendo el 10% representativo de cada libro.
    """
    # 1. Revisar si el índice ya existe
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        logger.info("El índice y los metadatos ya existen. Saliendo de build_index.")
        return

    # 2. Verificar libros existentes en la carpeta
    existing_books = [fname for fname in os.listdir(folder_path) if fname.endswith('.txt')]
    if len(existing_books) < MIN_BOOKS:
        faltan = MIN_BOOKS - len(existing_books)
        logger.info(f"Se requieren al menos {MIN_BOOKS} libros. Actualmente hay {len(existing_books)}. Descargando {faltan} libros más...")
        download_books_from_internet_archive(max_books=faltan)
    
    # 3. Cargar documentos
    logger.info(f"Cargando documentos desde: {folder_path}")
    documents = load_documents_from_folder(folder_path)
    
    if not documents:
        logger.error("No se encontraron documentos para indexar después de la descarga.")
        return
    
    # 4. Crear chunks y metadatos
    chunks = []
    chunk_metadata = []
    for doc_id, doc in enumerate(documents):
        doc_chunks = chunk_text(doc, max_sentences=2)
        if not doc_chunks:
            continue
        for chunk in doc_chunks:
            chunks.append(chunk)
            chunk_metadata.append({
                'document_id': doc_id,
                'text': chunk,
                'source': 'internet_archive'
            })
    
    if not chunks:
        logger.error("No se generaron chunks válidos para indexar.")
        return
    
    # 5. Generar embeddings y normalizarlos
    logger.info("Generando embeddings...")
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True)
        # Normalizar los embeddings para similitud coseno
        faiss.normalize_L2(chunk_embeddings)
    except Exception as e:
        logger.error(f"Error generando embeddings: {str(e)}")
        return

    # 6. Preparación del submuestreo para entrenamiento del índice.
    # Se agrupan los índices de los chunks por 'document_id'
    from collections import defaultdict
    doc_indices = defaultdict(list)
    for idx, meta in enumerate(chunk_metadata):
        doc_indices[meta['document_id']].append(idx)
    
    training_indices = []
    for doc_id, indices in doc_indices.items():
        n_sample = max(1, int(0.1 * len(indices)))  # tomo al menos 1
        sampled = random.sample(indices, n_sample)
        training_indices.extend(sampled)
    training_embeddings = chunk_embeddings[training_indices]
    
    logger.info(f"Se usarán {len(training_embeddings)} embeddings para entrenar el índice (de un total de {chunk_embeddings.shape[0]}).")
    
    # 7. Construir el índice FAISS con clustering (IndexIVFFlat)
    logger.info("Construyendo el índice FAISS con clustering (IndexIVFFlat)...")
    try:
        dimension = chunk_embeddings.shape[1]
        nlist = 100  # Número de clusters. Ajustable según tus datos
        
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        if not index.is_trained:
            logger.info("Entrenando el índice IVFFlat con los embeddings seleccionados...")
            index.train(training_embeddings)
        
        # Se añade el conjunto completo de embeddings al índice entrenado
        index.add(chunk_embeddings)
        norms = np.linalg.norm(chunk_embeddings, axis=1)
        logger.info(f"Normas de embeddings - Min: {norms.min():.4f}, Max: {norms.max():.4f}")
    except Exception as e:
        logger.error(f"Error construyendo índice FAISS: {str(e)}")
        return
    
    # 8. Guardar el índice y los metadatos en disco
    logger.info("Guardando el índice y los metadatos...")
    try:
        faiss.write_index(index, index_file)
        with open(metadata_file, 'wb') as f:
            pickle.dump(chunk_metadata, f)
    except Exception as e:
        logger.error(f"Error guardando archivos: {str(e)}")
        return
    
    logger.info(f"Índice generado con éxito. Total documentos: {len(documents)}, Total chunks: {len(chunks)}")

if __name__ == "__main__":
    start_time = time.time()
    build_index(BOOKS_FOLDER)
    logger.info(f"Proceso finalizado en {time.time() - start_time:.2f} segundos.")
