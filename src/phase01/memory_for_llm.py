from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# steps to follow
# . load raw pds
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_PATH = os.path.join(BASE_DIR, "data")  

def pdf_loader(path):
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents
documents = pdf_loader(path= DATA_PATH)

# . creat chunks
def create_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = text_splitter.split_documents(data)
    return chunks
chunks = create_chunks(data = documents)
# print(len(chunks))

# . create vector embeddings
def create_embeddings():
    embedding_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
embedding_model  = create_embeddings()

# . store embeddings in FAISS
DB_PATH="vectorstore/faiss_db"
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_PATH)