import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt', quiet=True)

def chunk_fixed_char(text, chunk_size=1000):
    print("fixed vhars")
    text = re.sub(r'\s+', ' ', text).strip()
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def chunk_fixed_tokens(text, chunk_size=200):
    print("fixed tokens")    
    tokens = word_tokenize(text)
    chunks = [' '.join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

def chunk_sentence_based(text, max_chars=1000):
    print("sentence based")    
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

def chunk_paragraph_based(text):
    print("paragraph based")    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    print("recursive")
    return paragraphs

def chunk_recursive(text, chunk_size=1000, separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]):
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

def chunk_sliding_window(text, window_size=800, overlap=200):
    print("sliding window")    
    text = re.sub(r'\n\s*\n', ' ', text).strip()
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + window_size, len(text))
        chunks.append(text[start:end])
        start += (window_size - overlap)
    
    return chunks


