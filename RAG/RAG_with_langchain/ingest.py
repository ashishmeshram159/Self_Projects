import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Define persistent storage directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_document(file_path):
    # Load document
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")

    docs = loader.load()
    
    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # Store in ChromaDB
    db = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=persistent_directory)
    db.persist()

    print(f" Document '{file_path}' ingested successfully!")

if __name__ == "__main__":
    file_path = input("Enter the path to the document (PDF/TXT): ")
    ingest_document(file_path)
