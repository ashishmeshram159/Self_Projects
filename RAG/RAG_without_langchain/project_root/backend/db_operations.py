
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer

# 1) Initialize Chroma DB client & collection
client = chromadb.Client(Settings())
collection = client.get_or_create_collection("pdf_chunks")

# 2) Load your embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3) Use the same tokenizer as your GPT-Neo model for consistent token-length checks
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def store_chunks(chunks, pdf_name):
    for idx, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk)
        # Generate a unique chunk ID (pdf_name + index)
        chunk_id = f"{pdf_name}_chunk_{idx}"

        metadata = {
            "pdf_name": pdf_name,
            "chunk_index": idx
        }

        collection.add(
            ids=[chunk_id],
            documents=[chunk],
            metadatas=[metadata],
            embeddings=[embedding]
        )

def retrieve_context(query, top_k=5, max_tokens_per_chunk=300):
    # 1) Encode the query
    query_embedding = embedding_model.encode(query)

    # 2) Query the vector store
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Flatten the list of documents
    docs = [doc for docs in results["documents"] for doc in docs]

    # 3) Truncate each document to max_tokens_per_chunk
    truncated_docs = []
    for doc in docs:
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        tokens = tokens[:max_tokens_per_chunk]
        truncated_text = tokenizer.decode(tokens)
        truncated_docs.append(truncated_text)

    return truncated_docs
