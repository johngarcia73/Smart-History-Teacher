from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt_tab')
# Prueba de tokenización en español
texto = "Hola mundo. Esto es una prueba."
print("Tokenización de oraciones:", sent_tokenize(texto, language='spanish'))
print("Tokenización de palabras:", word_tokenize(texto, language='spanish'))