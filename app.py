import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import traceback

import streamlit as st

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI

print("Current working directory:", os.getcwd())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "data.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = os.environ.get("GOOGLE_MODEL", "gemini-3.6-flash")
# Upper bound on retrieved chunks. The actual k used is min(this, total
# documents), so on a small catalog like this one every record is always
# retrieved and "list all products/orders/customers" style questions get
# complete context instead of an arbitrary top-3 similarity slice.
MAX_RETRIEVER_K = 20

SYSTEM_PROMPT = (
    "You are a helpful assistant for an online store. Answer the user's "
    "question using ONLY the information in the context below. Be concise "
    "and specific (e.g. include exact prices, stock counts, statuses, or "
    "emails when asked). If the answer is not contained in the context, "
    "say you don't have that information — do not make anything up.\n\n"
    "Context:\n{context}"
)


class LangChainRAGChat:
    """RAG pipeline built on LangChain: FAISS vector store for retrieval,
    a HuggingFace sentence-transformer for embeddings, and an LLM
    (via a retrieval chain) for actually generating the answer from the
    retrieved context."""

    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.embedding_model = None
        self.vectorstore = None
        self.retrieval_chain = None

    def _build_documents(self):
        """Turn each record into a LangChain Document with useful metadata,
        instead of an opaque formatted string. This lets the retriever and
        any future tool/agent logic reason about record type and id."""
        documents = []

        for product in self.data.get("products", []):
            content = (
                f"Product {product['id']}: {product['name']} - "
                f"{product['description']} - Category: {product['category']} - "
                f"Price: ${product['price']} - Stock: {product['stock']}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={"type": "product", "id": product["id"], "name": product["name"]},
                )
            )

        for order in self.data.get("orders", []):
            content = (
                f"Order {order['id']}: Customer {order['customer_name']} "
                f"ordered on {order['date']} - Total: ${order['total']} - "
                f"Status: {order['status']}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={"type": "order", "id": order["id"], "customer": order["customer_name"]},
                )
            )

        for customer in self.data.get("customers", []):
            content = (
                f"Customer {customer['id']}: {customer['name']} - "
                f"Email: {customer['email']} - Joined: {customer['join_date']} - "
                f"Total Orders: {customer['total_orders']}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={"type": "customer", "id": customer["id"], "name": customer["name"]},
                )
            )

        return documents

    def setup(self):
        print("Loading HuggingFace embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Embedding model loaded OK")

        documents = self._build_documents()
        print(f"Documents created: {len(documents)}")

        print("Building FAISS vector store...")
        self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        print(f"FAISS index has {self.vectorstore.index.ntotal} documents")

        retriever_k = min(MAX_RETRIEVER_K, len(documents))
        print(f"Retriever k = {retriever_k}")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": retriever_k})

        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and export it before "
                "running the app, e.g. `export GOOGLE_API_KEY=...` (see README)."
            )

        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        self.retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    def answer_query(self, query):
        result = self.retrieval_chain.invoke({"input": query})
        answer = result["answer"]
        sources = [doc.page_content for doc in result.get("context", [])]
        return answer, sources


def main():
    st.title("💬 Store Data Assistant")
    st.caption("Ask questions about products, orders, and customers")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chat" not in st.session_state:
        st.session_state.rag_chat = None
    if "load_error" not in st.session_state:
        st.session_state.load_error = None

    if st.session_state.rag_chat is None and st.session_state.load_error is None:
        with st.spinner("Loading models... This might take a minute..."):
            try:
                rag = LangChainRAGChat(JSON_PATH)
                rag.setup()
                st.session_state.rag_chat = rag
                print("RAG chat initialized successfully")
            except Exception:
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
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                    traceback.print_exc()


if __name__ == "__main__":
    main()