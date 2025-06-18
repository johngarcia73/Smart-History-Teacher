import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt', quiet=True)

# 1. Estrategia Actual: Fixed-size por caracteres (baseline)
def chunk_fixed_char(text, chunk_size=1000):
    print("fixed vhars")
    """Chunking por tamaño fijo de caracteres"""
    text = re.sub(r'\s+', ' ', text).strip()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 2. Fixed-size por tokens (más semántico)
def chunk_fixed_tokens(text, chunk_size=200):
    print("fixed tokens")    
    """Chunking por número fijo de tokens (aproximadamente 200 tokens ≈ 1000 caracteres)"""
    tokens = word_tokenize(text)
    chunks = [' '.join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

# 3. Basado en oraciones completas
def chunk_sentence_based(text, max_chars=1000):
    print("sentence based")    
    """Reseta límites naturales de oraciones manteniendo tamaño máximo"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# 4. Basado en párrafos naturales
def chunk_paragraph_based(text):
    print("paragraph based")    
    """Usa saltos de línea naturales para chunks de párrafos"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    print("recursive")
    return paragraphs

# 5. Recursive Character Text Splitting (avanzado)
def chunk_recursive(text, chunk_size=1000, separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]):
    """División recursiva usando jerarquía de separadores"""
    if len(text) <= chunk_size:
        return [text]
    
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            break
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        if len(current_chunk) + len(part) + len(sep) <= chunk_size:
            current_chunk += part + sep
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = part + sep
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Si algún chunk sigue siendo muy grande, recursión
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size:
            final_chunks.extend(chunk_recursive(chunk, chunk_size, separators))
        else:
            final_chunks.append(chunk)
    
    return final_chunks

# 6. Sliding Window con overlap (para RAG)
def chunk_sliding_window(text, window_size=800, overlap=200):
    print("sliding window")    
    """Chunks con solapamiento para mantener contexto"""
    text = re.sub(r'\n\s*\n', ' ', text).strip()
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + window_size, len(text))
        chunks.append(text[start:end])
        start += (window_size - overlap)
    
    return chunks


