import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load API keys from .env
load_dotenv()

# Define persistent storage directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Initialize embeddings and load ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define retriever with MMR search for better recall
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o")

# Contextualize question prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and latest user question, "
               "reformulate it into a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# QA prompt template
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Use retrieved context to answer the question "
               "in at most three sentences. If unsure, say 'I don't know'.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create RAG pipeline
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_rag_response(user_query, chat_history):
    return rag_chain.invoke({"input": user_query, "chat_history": chat_history})

