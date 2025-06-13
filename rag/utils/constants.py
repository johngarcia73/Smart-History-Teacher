INITIATOR_JID = "initiator@localhost"
SEARCH_JID = "search_agent@localhost"
EVAL_JID = "eval_agent@localhost"
PROMPT_JID = "prompt_agent@localhost"
SCRAPER_JID = "scrapper_agent@localhost"
CRAWLER_JID = "crawler_agent@localhost"   

INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.pickle"

SERVER = "localhost"
PASSWORDS = {
    INITIATOR_JID: "notelasabes",
    SEARCH_JID: "notelasabes",
    EVAL_JID: "notelasabes",
    PROMPT_JID: "notelasabes",
    SCRAPER_JID: "notelasabes",
    CRAWLER_JID: "notelasabes"
}

INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.pickle"
DOCUMENTS_FOLDER = "./data"

HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

SCORE_THRESHOLD = 7
WIKIPEDIA_API = "https://es.wikipedia.org/w/api.php"

CONFIDENCE_THRESHOLD = 0.8  # Ajustar según evaluación
