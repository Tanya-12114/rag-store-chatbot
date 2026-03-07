import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import os

print("Current working directory:", os.getcwd())

# Define the path to the JSON data — use a relative path so it works on any machine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, 'data.json')

# Load data from the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Display the data for debugging purposes
print(data)


class SimpleRAGChat:
    def __init__(self, json_path):
        # Load data from the given JSON path
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Initialize the documents attribute
        self.documents = []  # Ensure documents is initialized before other methods
        
        # Initialize models
        self.init_models()

        # Create the document store after initialization
        self.create_document_store()

    def init_models(self):
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch
        
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    def create_document_store(self):
        # Ensure that self.documents is initialized before appending to it
        if not hasattr(self, 'documents'):
            self.documents = []

        # Add product information
        for product in self.data.get('products', []):
            doc = f"Product {product['id']}: {product['name']} - {product['description']} - Category: {product['category']} - Price: ${product['price']} - Stock: {product['stock']}"
            self.documents.append(doc)
    
        # Add order information (only if 'orders' key exists)
        orders = self.data.get('orders', [])
        if orders:
            for order in orders:
                doc = f"Order {order['id']}: Customer {order['customer_name']} ordered on {order['date']} - Total: ${order['total']} - Status: {order['status']}"
                self.documents.append(doc)
        else:
            print("No 'orders' key found in the data!")

        # Add customer information (if present)
        for customer in self.data.get('customers', []):
            doc = f"Customer {customer['id']}: {customer['name']} - Email: {customer['email']} - Joined: {customer['join_date']} - Total Orders: {customer['total_orders']}"
            self.documents.append(doc)

        # Debugging: Check if documents are created
        print(f"Documents created: {len(self.documents)} documents.")
        
        # Guard: FAISS will crash if there are no documents to index
        if not self.documents:
            raise ValueError("No documents were created from the data. Check data.json for valid 'products', 'orders', or 'customers' keys.")

        # Create FAISS index
        embeddings = self.embedding_model.encode(self.documents)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        # Debugging: Check if FAISS index has documents
        print(f"FAISS index has {self.index.ntotal} documents.")

    def retrieve_relevant_docs(self, query, k=2):
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        _, indices = self.index.search(query_embedding, k)
        
        # Return relevant documents
        return [self.documents[i] for i in indices[0]]
    
    def answer_query(self, query):
        relevant_docs = self.retrieve_relevant_docs(query)
        context = "\n".join(relevant_docs)
        
        prompt = f"""Given this context about our store:
        {context}
        Answer this question: {query}
        Provide a clear and concise answer."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_length=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, relevant_docs


# Streamlit application
def main():
    st.title("💬 Store Data Assistant")
    st.caption("Ask questions about products, orders, and customers")
    
    # Initialize session state for storing messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize RAG chat — use relative path, not a hardcoded Windows path
    if 'rag_chat' not in st.session_state:
        with st.spinner("Loading models... This might take a minute..."):
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(BASE_DIR, 'data.json')
            st.session_state.rag_chat = SimpleRAGChat(json_path)

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show sources if available
            if "sources" in message:
                with st.expander("View sources"):
                    for source in message["sources"]:
                        st.write(source)
    
    # Get user input for the query
    if prompt := st.chat_input("Ask about the store data..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from the assistant
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.rag_chat.answer_query(prompt)
                
                st.write(answer)
                
                # Save assistant response with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

if __name__ == "__main__":
    main()