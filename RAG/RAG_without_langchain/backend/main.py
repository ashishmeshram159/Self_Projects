from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import shutil
import os.path

# Import from your local modules
from pdf_processing import process_pdf
from db_operations import store_chunks, retrieve_context

# Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the open-source language model
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure pad_token is set (some GPT-Neo checkpoints need this)
tokenizer.pad_token = tokenizer.eos_token

# Initialize FastAPI app
app = FastAPI()
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure the directory exists

# CORS middleware (if you need cross-origin requests from a frontend)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a schema for the query request
class QueryRequest(BaseModel):
    query: str

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the PDF into chunks
    chunks = process_pdf(file_path)

    # Use the PDF filename (minus extension) as a unique identifier
    pdf_name = os.path.splitext(file.filename)[0]

    # Store chunks in the DB with the unique PDF name
    store_chunks(chunks, pdf_name)

    return {"message": "PDF processed and stored in Chroma DB successfully."}

@app.post("/ask/")
async def ask_query(query_request: QueryRequest):
    query = query_request.query

    # Retrieve up to 5 chunks, each truncated to 300 tokens
    context_chunks = retrieve_context(query, top_k=5, max_tokens_per_chunk=300)
    combined_context = "\n".join(context_chunks)

    # Construct a prompt
    prompt = f"Context:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"

    # GPT-Neo 1.3B typically has a max_position_embeddings of 2048
    max_total_length = model.config.max_position_embeddings
    max_new_tokens = 150

    # We want to leave room for 150 tokens of generation
    prompt_max_length = max_total_length - max_new_tokens

    # Tokenize with truncation so prompt + new tokens won't exceed 2048:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_max_length,
        padding=True
    )

    # Generate the answer
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,  
    )

    # Decode the result
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "query": query,
        "context": combined_context,
        "message": response_text,
    }


# # Use this to run the FastAPI app while in the 'backend' directory:
# uvicorn main:app --reload