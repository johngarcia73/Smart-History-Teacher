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

# Configuración: carpeta donde se guardarán los libros descargados
BOOKS_FOLDER = "data"
os.makedirs(BOOKS_FOLDER, exist_ok=True)
MIN_BOOKS = 11

# Palabras clave para verificar que es historia
HISTORY_KEYWORDS = {
    "historia", "revolución", "guerra", "independencia", 
    "colonial", "imperio", "renacimiento", "conquista"
}


def search_gutenberg_books(count):
    """
    Busca libros en español con temática histórica usando la API Gutendex.
    Devuelve una lista de dicts: {id, title}.
    """
    books = []
    page = 1
    while len(books) < 4:
        resp = requests.get(
            "https://gutendex.com/books",
            params={
                "languages": "es",
                "topic": "history",
                "page": page
            },
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("results", []):
            title = item.get("title", "")
            # Verificamos si el título contiene alguna clave de historia
            books.append({
                "source": "GUTENBERG",
                "id": item["id"],
                "title": title
            })
            if len(books) >= count:
                break
        if not data.get("next"):
            break
        print("7")
        page += 1
    return books[:count]


def download_gutenberg_book(book):
    """
    Descarga el texto del libro de Gutenberg. Intenta dos sufijos comunes.
    Devuelve la ruta al archivo .txt o None.
    """
    identifier = book["id"]
    filename = os.path.join(BOOKS_FOLDER, f"GUT_{identifier}.txt")

    if os.path.exists(filename):
        return filename
    print("1")

    for suffix in ("-0.txt", "-8.txt"):
        print("2")
        url = f"https://www.gutenberg.org/files/{identifier}/{identifier}{suffix}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 5000:
                # Verificamos contenido histórico mínimo
                print("3")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(r.text)
                    print("4")
                return filename
        except Exception:
            continue
    return None


def search_archive_books(count):
    """
    Busca en Internet Archive libros con subject:History, idioma español.
    Devuelve lista de dicts: {id, title}.
    """
    params = {
        "q": 'subject:"History" AND language:(spa)',
        "fl[]": ["identifier", "title"],
        "sort[]": "downloads desc",
        "rows": str(count),
        "output": "json"
    }
    r = requests.get(
        "https://archive.org/advancedsearch.php",
        params=params,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    r.raise_for_status()
    docs = r.json()["response"]["docs"]
    books = []
    for d in docs:
        title = d.get("title", d["identifier"])
        if any(kw in title.lower() for kw in HISTORY_KEYWORDS):
            books.append({
                "source": "ARCHIVE",
                "id": d["identifier"],
                "title": title
            })
    return books[:count]


def download_archive_book(book):
    """
    Descarga el mayor .txt disponible de Internet Archive para el identificador dado.
    Devuelve la ruta al archivo o None.
    """
    identifier = book["id"]
    filename = BOOKS_FOLDER / f"IA_{identifier}.txt"
    if filename.exists():
        return filename

    # Obtén metadata y selecciona el .txt más grande
    meta = requests.get(f"https://archive.org/metadata/{identifier}", timeout=15).json()
    txts = [
        f for f in meta.get("files", [])
        if f.get("name", "").lower().endswith(".txt")
           and int(f.get("size", 0)) > 10240
    ]
    if not txts:
        return None
    largest = max(txts, key=lambda f: int(f.get("size", 0)))
    url = f"https://archive.org/download/{identifier}/{largest['name']}"

    # Descarga en streaming
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(8_192):
                f.write(chunk)

    # Verificamos contenido histórico
    snippet = filename.read_text(encoding="utf-8", errors="ignore")[:20_000].lower()
    if not any(kw in snippet for kw in HISTORY_KEYWORDS):
        filename.unlink(missing_ok=True)
        return None
    return filename


def download_history_collection(min_books=MIN_BOOKS):
    """
    Descarga hasta min_books combinando Gutenberg y Archive.
    """
    downloaded = []
    remaining = min_books

    print("0")
    #Gutenberg
    gut_books = search_gutenberg_books(remaining)
    for b in gut_books:
        if remaining <= 0:
            break
        path = download_gutenberg_book(b)
        if path:
            downloaded.append(path)
            remaining -= 1
            print(f"[GUT] {b['title']} -> {path.name}")
            time.sleep(0.2)

    #Internet Archive
    if remaining > 0:
        ia_books = search_archive_books(remaining)
        for b in ia_books:
            if remaining <= 0:
                break
            path = download_archive_book(b)
            if path:
                downloaded.append(path)
                remaining -= 1
                print(f"[IA] {b['title']} -> {path.name}")
                time.sleep(0.2)

    print(f"Total descargados: {len(downloaded)}")
    return downloaded



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

"""
def chunk_text(text, max_sentences=2):
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
"""
def chunk_text(text, chunk_size=1000): # Mas eficiente
    #text = text.replace("\n", " ")
    # Eliminar espacios innecesarios
    #text = " ".join(text.split())
    #chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    #return chunks
    return chunk_sentence_based(text)

def build_index(folder_path, index_file='faiss_index.bin', metadata_file='chunk_metadata.pickle'):
    # 1. Revisar si el índice ya existe
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        logger.info("El índice y los metadatos ya existen. Saliendo de build_index.")
        return

    # 2. Verificar libros existentes en la carpeta
    existing_books = [fname for fname in os.listdir(folder_path) if fname.endswith('.txt')]
    if len(existing_books) < MIN_BOOKS:
        faltan = MIN_BOOKS - len(existing_books)
        logger.info(f"Se requieren al menos {MIN_BOOKS} libros. Actualmente hay {len(existing_books)}. Descargando {faltan} libros más...")
        download_history_collection(faltan)
    
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
        # Usamos el método rápido para dividir el documento en chunks
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
    
    # 5. Generar embeddings y normalizarlos (con batch_size para acelerar)
    logger.info("Generando embeddings...")
    try:
        #embedder = SentenceTransformer('all-MiniLM-L6-v2')
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

        print("1")
        chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True, batch_size=512)
        print("2")
        # Normalizar los embeddings para similitud coseno
        faiss.normalize_L2(chunk_embeddings)
        print("3")
    except Exception as e:
        logger.error(f"Error generando embeddings: {str(e)}")
        return


    # 6. Preparación del submuestreo para entrenamiento del índice.
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
