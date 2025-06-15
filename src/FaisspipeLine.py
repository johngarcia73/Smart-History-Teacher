# pipelines.py
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class FaissPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
        self.knowledge_base = FAISS.load_local("historia.faiss", self.embeddings)

    def process_item(self, item, spider):
        doc = Document(
            page_content=item['content'],
            metadata={
                'titulo': item['title'],
                'fuente': item['url'],
                'fecha_ingreso': datetime.now()
            }
        )
        # Verificación de duplicados por embedding
        if not self.is_duplicate(doc.page_content):
            self.knowledge_base.add_documents([doc])
            self.knowledge_base.save_local("historia.faiss")

    def is_duplicate(self, text, threshold=0.95):
        query_embedding = self.embeddings.embed_query(text)
        similar_docs = self.knowledge_base.similarity_search_by_vector(query_embedding, k=1)
        return len(similar_docs) > 0 and cosine_similarity(query_embedding, similar_docs[0].embedding) > threshold
    