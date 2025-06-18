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
from bs4 import BeautifulSoup
import re
from utils.logging import configure_logging
from utils.chunking import chunk_fixed_char, chunk_fixed_tokens, chunk_paragraph_based, chunk_recursive, chunk_sentence_based, chunk_sliding_window


# Configurar logging
logger = logging.getLogger(__name__)

# carpeta donde se guardarán los libros descargados
BOOKS_FOLDER = "data"
os.makedirs(BOOKS_FOLDER, exist_ok=True)
MIN_BOOKS = 1


HISTORY_KEYWORDS = [
    "historia", "revolución", "guerra", "independencia", "colonial", "imperio", 
    "edad media", "renacimiento", "reconquista", "descubrimiento", "conquista"
]

ACADEMIC_SOURCES = {
    "REDIAL": "https://redial-redae.eu/search?query=historia&format=book&lang=es",
    "HISTORIA_NACIONAL": "https://historianacional.cl/biblioteca?categoria=historia&formato=digital",
    "BIBLIOTECA_HISTORICA": "https://bibliotecahistorica.es/catalogo?tema=historia&idioma=es"
}




def search_history_books(source, count):
    """Búsqueda especializada en libros de historia en español"""
    logger.info(f"Buscando {count} libros de historia en {source}")
    books = []
    
    try:
        if source == "REDIAL":
            url = ACADEMIC_SOURCES[source]
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for item in soup.select('.result-item')[:count]:
                title = item.select_one('h3.title').get_text().strip()
                if not any(kw in title.lower() for kw in HISTORY_KEYWORDS):
                    continue
                book_id = item.select_one('a')['href'].split('/')[-1]
                books.append({
                    "source": source,
                    "id": book_id,
                    "title": title
                })
                
        elif source == "HISTORIA_NACIONAL":
            url = ACADEMIC_SOURCES[source]
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for item in soup.select('.book-card')[:count]:
                title = item.select_one('h4').get_text().strip()
                if not any(kw in title.lower() for kw in HISTORY_KEYWORDS):
                    continue
                book_id = item.select_one('a')['href'].split('/')[-1]
                books.append({
                    "source": source,
                    "id": book_id,
                    "title": title
                })
                
        elif source == "BIBLIOTECA_HISTORICA":
            url = ACADEMIC_SOURCES[source]
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for item in soup.select('.catalog-item')[:count]:
                title = item.select_one('h3').get_text().strip()
                if not any(kw in title.lower() for kw in HISTORY_KEYWORDS):
                    continue
                book_id = item.select_one('a')['href'].split('/')[-1]
                books.append({
                    "source": source,
                    "id": book_id,
                    "title": title
                })
                
        logger.info(f"Encontrados {len(books)} libros de historia en {source}")
        return books
        
    except Exception as e:
        logger.error(f"Error buscando en {source}: {str(e)}")
        return []

def download_history_book(source, identifier, title):
    file_path = os.path.join(BOOKS_FOLDER, f"{source}_{identifier}.txt")
    
    if os.path.exists(file_path):
        return file_path
    
    try:
        if source == "REDIAL":
            url = f"https://redial-redae.eu/document/{identifier}/fulltext"
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.select_one('.document-content').get_text()
            
        elif source == "HISTORIA_NACIONAL":
            url = f"https://historianacional.cl/documento/{identifier}"
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.select_one('.book-content').get_text()
            
        elif source == "BIBLIOTECA_HISTORICA":
            url = f"https://bibliotecahistorica.es/documento/{identifier}"
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.select_one('.document-text').get_text()
        
        if not any(kw in text.lower() for kw in HISTORY_KEYWORDS[:5]):
            logger.warning(f"El contenido no parece histórico: {title}")
            return None
            
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 10000:
            logger.warning(f"Contenido demasiado corto: {title}")
            return None
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        return file_path
        
    except Exception as e:
        logger.error(f"Error descargando {title}: {str(e)}")
        return None

def download_history_collection(min_books):
    """Descarga libros de historia de fuentes especializadas"""
    downloaded = []
    remaining = min_books
    
    sources_order = ["HISTORIA_NACIONAL", "BIBLIOTECA_HISTORICA", "REDIAL"]
    
    for source in sources_order:
        if remaining <= 0:
            break
            
        books = search_history_books(source, remaining * 2)  # Buscar más para filtrar
        logger.info(f"Libros potenciales encontrados: {len(books)}")
        
        for book in books:
            if remaining <= 0:
                break
                
            # Priorizar libros con palabras clave en título
            title_score = sum(1 for kw in HISTORY_KEYWORDS if kw in book["title"].lower())
            if title_score < 2:
                continue
                
            file_path = download_history_book(
                source=book["source"],
                identifier=book["id"],
                title=book["title"]
            )
            
            if file_path:
                downloaded.append(file_path)
                remaining -= 1
                logger.info(f"Libro histórico descargado: {book['title']} ({source})")
                time.sleep(0.5)
    
    # Si aún faltan libros, usar búsqueda especializada en Internet Archive
    if remaining > 0:
        logger.info(f"Descargando {remaining} libros de respaldo")
        downloaded += download_history_from_archive(max_books=remaining)
    
    return downloaded

def download_history_from_archive(max_books=10):
    """Descarga específicamente libros de historia de Internet Archive"""
    logger.info(f"Descargando {max_books} libros de historia desde Internet Archive")
    
    params = {
        'q': 'subject:"History" AND title:historia AND language:(spa) AND mediatype:texts',
        'fl[]': ['identifier', 'title'],
        'sort[]': 'downloads desc',
        'rows': str(max_books),
        'output': 'json'
    }
    
    try:
        response = requests.get(
            "https://archive.org/advancedsearch.php",
            params=params,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        data = response.json()
        books = []
        
        for doc in data.get('response', {}).get('docs', [])[:max_books]:
            if any(kw in doc.get('title', '').lower() for kw in HISTORY_KEYWORDS):
                books.append({
                    'source': 'INTERNET_ARCHIVE',
                    'id': doc['identifier'],
                    'title': doc.get('title', doc['identifier'])
                })
        
        logger.info(f"Libros históricos encontrados en IA: {len(books)}")
        return [download_ia_book(book) for book in books if download_ia_book(book)]
        
    except Exception as e:
        logger.error(f"Error en búsqueda especializada: {str(e)}")
        return []
    
    
def download_ia_book(book_info):
    identifier = book_info['id']
    title = book_info['title']
    file_path = os.path.join(BOOKS_FOLDER, f"IA_{identifier}.txt")
    
    if os.path.exists(file_path):
        return file_path
    
    try:
        # Obtener metadatos del libro
        meta_url = f"https://archive.org/metadata/{identifier}"
        meta_response = requests.get(meta_url, timeout=30)
        meta_data = meta_response.json()
        
        # Encontrar el archivo de texto más grande
        txt_files = [f for f in meta_data.get('files', []) 
                    if f.get('name', '').lower().endswith('.txt') 
                    and int(f.get('size', '0')) > 10240]
        
        if not txt_files:
            logger.warning(f"No se encontraron archivos .txt válidos en {identifier}")
            return None
        
        largest_file = max(txt_files, key=lambda x: int(x['size']))
        file_name = largest_file['name']
        download_url = f"https://archive.org/download/{identifier}/{file_name}"
        
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read(20000)
            
        if not any(kw in text.lower() for kw in HISTORY_KEYWORDS[:5]):
            logger.warning(f"El contenido no parece histórico: {title}")
            os.remove(file_path)
            return None
        
        return file_path
        
    except Exception as e:
        logger.error(f"Error descargando {title} de IA: {str(e)}")
        return None
    
    

def load_documents_from_folder(folder_path):
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


def chunk_text(text, chunk_size=1000): # Mas eficiente
    #text = text.replace("\n", " ")
    #text = " ".join(text.split())
    #chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    #return chunks
    return chunk_sentence_based(text)

def build_index(folder_path, index_file='faiss_index.bin', metadata_file='chunk_metadata.pickle'):
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        logger.info("El índice y los metadatos ya existen. Saliendo de build_index.")
        return

    existing_books = [fname for fname in os.listdir(folder_path) if fname.endswith('.txt')]
    if len(existing_books) < MIN_BOOKS:
        faltan = MIN_BOOKS - len(existing_books)
        logger.info(f"Se requieren al menos {MIN_BOOKS} libros. Actualmente hay {len(existing_books)}. Descargando {faltan} libros más...")
        download_history_collection(faltan)
    
    logger.info(f"Cargando documentos desde: {folder_path}")
    documents = load_documents_from_folder(folder_path)
    
    if not documents:
        logger.error("No se encontraron documentos para indexar después de la descarga.")
        return
    
    chunks = []
    chunk_metadata = []
    for doc_id, doc in enumerate(documents):
        doc_chunks = chunk_text(doc, chunk_size=1000)
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
    
    logger.info("Generando embeddings...")
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("1")
        chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True, batch_size=256)
        print("2")
        faiss.normalize_L2(chunk_embeddings)
        print("3")
    except Exception as e:
        logger.error(f"Error generando embeddings: {str(e)}")
        return


    doc_indices = defaultdict(list)
    for idx, meta in enumerate(chunk_metadata):
        doc_indices[meta['document_id']].append(idx)
    
    training_indices = []
    for doc_id, indices in doc_indices.items():
        n_sample = max(1, int(0.1 * len(indices)))  # al menos 1
        sampled = random.sample(indices, n_sample)
        training_indices.extend(sampled)
    training_embeddings = chunk_embeddings[training_indices]
    
    logger.info(f"Se usarán {len(training_embeddings)} embeddings para entrenar el índice (de un total de {chunk_embeddings.shape[0]}).")
    
    # 7. Construir el índice FAISS con clustering (IndexIVFFlat)
    logger.info("Construyendo el índice FAISS con clustering (IndexIVFFlat)...")
    try:
        dimension = chunk_embeddings.shape[1]
        nlist = 100  # Número de clusters
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.make_direct_map()
        
        if not index.is_trained:
            logger.info("Entrenando el índice IVFFlat con los embeddings seleccionados...")
            index.train(training_embeddings)
        
        # Agregar todo el conjunto de embeddings al índice entrenado
        index.add(chunk_embeddings)
        norms = np.linalg.norm(chunk_embeddings, axis=1)
        logger.info(f"Normas de embeddings - Min: {norms.min():.4f}, Max: {norms.max():.4f}")
    except Exception as e:
        logger.error(f"Error construyendo índice FAISS: {str(e)}")
        return
    
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
