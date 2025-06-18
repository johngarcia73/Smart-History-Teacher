import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.chunking import chunk_fixed_char, chunk_fixed_tokens, chunk_paragraph_based, chunk_recursive, chunk_sentence_based, chunk_sliding_window
from sentence_transformers import SentenceTransformer
import logging
from indexer import load_documents_from_folder
from utils.constants import DOCUMENTS_FOLDER
import re  # Asegúrate de importar re para las operaciones regex
from utils.logging import configure_logging

def evaluate_chunking_strategies(documents, strategies):
    """Evalúa múltiples estrategias de chunking en una lista de documentos"""
    all_results = []
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    for doc_id, document in enumerate(documents):
        doc_results = {
            'doc_id': doc_id,
            'doc_length': len(document),
            'strategies': {}
        }
        
        for name, strategy in strategies.items():
            try:
                # 1. Métrica: Tiempo de ejecución
                start_time = time.perf_counter()
                chunks = strategy(document)
                duration = time.perf_counter() - start_time
                
                # 2. Métrica: Distribución de tamaños
                lengths = [len(chunk) for chunk in chunks]
                
                # 3. Métrica: Coherencia semántica
                coherence = 0
                """
                if len(chunks) > 1:
                    embeddings = embedder.encode(chunks)
                    similarities = cosine_similarity(embeddings)
                    np.fill_diagonal(similarities, 0)
                    coherence = np.mean(similarities) if similarities.size > 0 else 0
                """
                
                # 4. Métrica: Pérdida de información
                reconstructed = " ".join(chunks)
                info_loss = abs(len(document) - len(reconstructed)) / len(document) if len(document) > 0 else 0
                
                doc_results['strategies'][name] = {
                    'time': duration,
                    'num_chunks': len(chunks),
                    'avg_length': np.mean(lengths) if lengths else 0,
                    'length_std': np.std(lengths) if len(lengths) > 1 else 0,
                    'coherence': coherence,
                    'info_loss': info_loss,
                    'chunks_sample': chunks[:3]  # Muestra primeros chunks
                }
                
            except Exception as e:
                logger.error(f"Error en estrategia {name} con documento {doc_id}: {str(e)}")
                doc_results['strategies'][name] = None
        
        all_results.append(doc_results)
        print("Documento terminado.")
    
    return all_results

# SOLUCIÓN: Definición de estrategias sin partial
# Usamos funciones wrapper para pasar los parámetros
strategies = {
    "Sliding Window": lambda text: chunk_sliding_window(text, window_size=800, overlap=200),
    "Fixed Char": lambda text: chunk_fixed_char(text, chunk_size=1000),
    "Fixed Tokens": lambda text: chunk_fixed_tokens(text, chunk_size=200),
    "Sentence-based": lambda text: chunk_sentence_based(text, max_chars=1000),
    #"Paragraph-based": chunk_paragraph_based,
    #"Recursive": lambda text: chunk_recursive(text, chunk_size=1000)
}

current_log = configure_logging()
logger = logging.getLogger(__name__)

# Cargar documentos
logger.info(f"Cargando documentos desde: {DOCUMENTS_FOLDER}")
documents = load_documents_from_folder(DOCUMENTS_FOLDER)

# Verificar que tenemos documentos antes de continuar
if not documents:
    logger.error("No se encontraron documentos para procesar")
    exit()

# Evaluar estrategias en todos los documentos
results = evaluate_chunking_strategies(documents, strategies)

# Calcular estadísticas agregadas
strategy_stats = {}

# Inicializar estructura para estadísticas
for strategy_name in strategies.keys():
    strategy_stats[strategy_name] = {
        'times': [],
        'num_chunks': [],
        'coherences': [],
        'info_losses': []
    }











# Recolectar datos de todos los documentos
for doc_result in results:
    for strategy_name, metrics in doc_result['strategies'].items():
        if metrics:
            strategy_stats[strategy_name]['times'].append(metrics['time'])
            strategy_stats[strategy_name]['num_chunks'].append(metrics['num_chunks'])
            #strategy_stats[strategy_name]['coherences'].append(metrics['coherence'])
            strategy_stats[strategy_name]['info_losses'].append(metrics['info_loss'])

# Calcular promedios y mostrar resultados
logger.info("\n=== RESULTADOS AGREGADOS ===")
logger.info(f"{'Estrategia':<20} | {'Tiempo (ms)':>12} | {'Chunks/Doc':>10} | {'Coherencia':>10} | {'Pérdida (%)':>12}")
logger.info("-" * 75)

for strategy_name, stats in strategy_stats.items():
    if stats['times']:  # Solo si hay datos
        avg_time = np.mean(stats['times']) * 1000  # ms
        avg_chunks = np.mean(stats['num_chunks'])
        #avg_coherence = np.mean(stats['coherences'])
        avg_info_loss = np.mean(stats['info_losses']) * 100  # Porcentaje
        
        logger.info(f"{strategy_name:<20} | {avg_time:>12.2f} | {avg_chunks:>10.1f}  | {avg_info_loss:>12.2f}")
        #| {avg_coherence:>10.4f}
    else:
        logger.warning(f"{strategy_name:<20} | Sin datos válidos")

# Resultados detallados por documento (opcional)
"""
logger.info("\n=== RESUMEN POR DOCUMENTO ===")
for doc_result in results:
    logger.info(f"\nDocumento ID: {doc_result['doc_id']} - Longitud: {doc_result['doc_length']} caracteres")
    for strategy_name, metrics in doc_result['strategies'].items():
        if metrics:
            logger.info(f"  {strategy_name}: {metrics['num_chunks']} chunks, "
                        f"tiempo: {metrics['time']*1000:.2f}ms, "
                        f"coherencia: {metrics['coherence']:.4f}, "
                        f"pérdida: {metrics['info_loss']*100:.2f}%")
"""