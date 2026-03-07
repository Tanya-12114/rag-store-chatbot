import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import streamlit as st
import json
import traceback
import numpy as np

print("Current working directory:", os.getcwd())

# Define the path to the JSON data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, 'data.json')

with open(json_path, 'r') as f:
    data = json.load(f)

print("Data loaded successfully")


class SimpleRAGChat:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.documents = []
        self.embedding_model = None
        self.index = None

    def init_models(self):
        print("Loading SentenceTransformer...")
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("SentenceTransformer loaded OK")

    def create_document_store(self):
        import faiss

        self.documents = []

        for product in self.data.get('products', []):
            doc = f"Product {product['id']}: {product['name']} - {product['description']} - Category: {product['category']} - Price: ${product['price']} - Stock: {product['stock']}"
            self.documents.append(doc)

        orders = self.data.get('orders', [])
        if orders:
            for order in orders:
                doc = f"Order {order['id']}: Customer {order['customer_name']} ordered on {order['date']} - Total: ${order['total']} - Status: {order['status']}"
                self.documents.append(doc)

        for customer in self.data.get('customers', []):
            doc = f"Customer {customer['id']}: {customer['name']} - Email: {customer['email']} - Joined: {customer['join_date']} - Total Orders: {customer['total_orders']}"
            self.documents.append(doc)

        print(f"Documents created: {len(self.documents)}")

        embeddings = self.embedding_model.encode(self.documents)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings))
        print(f"FAISS index has {self.index.ntotal} documents")

    def setup(self):
        self.init_models()
        self.create_document_store()

    def retrieve_relevant_docs(self, query, k=3):
        query_embedding = self.embedding_model.encode([query])
        _, indices = self.index.search(np.array(query_embedding), k)
        return [self.documents[i] for i in indices[0]]

    def answer_query(self, query):
        relevant_docs = self.retrieve_relevant_docs(query)
        answer = self._build_answer(query, relevant_docs)
        return answer, relevant_docs

    def _build_answer(self, query, docs):
        query_lower = query.lower()

        for doc in docs:
            doc_lower = doc.lower()

            # Price queries
            if any(word in query_lower for word in ['price', 'cost', 'how much']):
                if 'price:' in doc_lower:
                    parts = doc.split('Price: $')
                    if len(parts) > 1:
                        price = parts[1].split(' ')[0].split('-')[0].strip()
                        name = doc.split(':')[1].split('-')[0].strip()
                        return f"The price of {name} is ${price}."

            # Stock queries
            if any(word in query_lower for word in ['stock', 'available', 'inventory', 'many']):
                if 'stock:' in doc_lower:
                    parts = doc.split('Stock: ')
                    if len(parts) > 1:
                        stock = parts[1].strip()
                        name = doc.split(':')[1].split('-')[0].strip()
                        return f"{name} has {stock} units in stock."

            # Status queries
            if any(word in query_lower for word in ['status', 'shipped', 'pending', 'order']):
                if 'status:' in doc_lower:
                    parts = doc.split('Status: ')
                    if len(parts) > 1:
                        status = parts[1].strip()
                        order_id = doc.split(':')[0].replace('Order', '').strip()
                        return f"Order {order_id} status is: {status}."

            # Email queries
            if any(word in query_lower for word in ['email', 'contact', 'mail']):
                if 'email:' in doc_lower:
                    parts = doc.split('Email: ')
                    if len(parts) > 1:
                        email = parts[1].split(' ')[0].split('-')[0].strip()
                        name = doc.split(':')[1].split('-')[0].strip()
                        return f"{name}'s email is {email}."

        # Default: return most relevant document
        if docs:
            return f"Based on our store data: {docs[0]}"
        return "I couldn't find relevant information for your query."


def main():
    st.title("💬 Store Data Assistant")
    st.caption("Ask questions about products, orders, and customers")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_chat' not in st.session_state:
        st.session_state.rag_chat = None
    if 'load_error' not in st.session_state:
        st.session_state.load_error = None

    if st.session_state.rag_chat is None and st.session_state.load_error is None:
        with st.spinner("Loading models... This might take a minute..."):
            try:
                rag = SimpleRAGChat(json_path)
                rag.setup()
                st.session_state.rag_chat = rag
                print("RAG chat initialized successfully")
            except Exception as e:
                error_msg = traceback.format_exc()
                st.session_state.load_error = error_msg
                print(f"FATAL ERROR: {error_msg}")

    if st.session_state.load_error:
        st.error("Failed to load models. See error below:")
        st.code(st.session_state.load_error)
        if st.button("Retry"):
            st.session_state.load_error = None
            st.rerun()
        return

    if st.session_state.rag_chat is None:
        st.warning("Models are still loading...")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                with st.expander("View sources"):
                    for source in message["sources"]:
                        st.write(source)

    if prompt := st.chat_input("Ask about the store data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = st.session_state.rag_chat.answer_query(prompt)
                    st.write(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
                    traceback.print_exc()


if __name__ == "__main__":
    main()