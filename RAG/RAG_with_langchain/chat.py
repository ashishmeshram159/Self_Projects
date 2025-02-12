from langchain.memory import ConversationBufferMemory
from rag import get_rag_response

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def continual_chat():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print(" Exiting chat...")
            break
        # Fetch response from RAG pipeline
        chat_history = memory.load_memory_variables({})["chat_history"]
        response = get_rag_response(user_input, chat_history)

        # Display response
        print(f"AI: {response['answer']}\n")

        # Save conversation history
        memory.save_context({"input": user_input}, {"answer": response["answer"]})
