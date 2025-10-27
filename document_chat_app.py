import os, tempfile, shutil
from dotenv import load_dotenv
from typing import Dict, List, Tuple


import streamlit as st

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document


load_dotenv()  # load env vars

# Base URL
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL", "https://api.ai.it.cornell.edu")

# Vector Store
PERSIST_DIR = os.getenv("PERSIST_DIR", ".chroma")
COLLECTION = "assignment1"

# Embeddings + LLM (LiteLLM/OpenAI-compatible)
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai.text-embedding-3-large")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "openai.gpt-5-chat")  # or the class-provided model alias

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))
FETCH_K = int(os.getenv("FETCH_K", "20"))

# App
APP_TITLE = "Assignment 1 — PDF and TXT RAG Chat"

SYSTEM_PROMPT = """You are a helpful assistant answering only from the provided documents.
- If the answer is not in the documents, say you can't find the answer in the given documents.
- Cite each answer with source file name (and page if available).
- Give helpful summary pointer at the end.
- Be concise and precise.
"""


def load_documents(paths: List[str]):
    docs = []
    for pth in paths:
        ext = os.path.splitext(pth)[1].lower()
        if ext == ".pdf":
            docs.extend(PyPDFLoader(pth).load())
        else:
            docs.extend(TextLoader(pth, encoding="utf-8").load())
    return docs

def get_document_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_or_update_index(chunks, collection_name=COLLECTION):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vs.add_documents(chunks)
    vs.persist()
    return vs

def list_index_stats(collection_name=COLLECTION):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    try:
        count = vs._collection.count()
    except Exception:
        count = 0
    return {"collection": collection_name, "docs": count}

def _format_context(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f"{src}" if page is None else f"{src} (p.{page+1})"
        lines.append(f"[{tag}] {d.page_content}")
    return "\n\n".join(lines)

def get_retriever(collection_name=COLLECTION):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vs.as_retriever(search_type="mmr", k=TOP_K, fetch_k=FETCH_K)

def answer_question(question: str, chat_history: List[Dict], collection_name=COLLECTION):
    retriever = get_retriever(collection_name)
    try:
        docs = retriever.invoke(question)
    except AttributeError:
        docs = retriever.get_relevant_documents(question)

    context = _format_context(docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Context documents:\n\n{context}"),
    ]

    for turn in chat_history[-4:]:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(SystemMessage(content=f"Assistant (previous): {turn['assistant']}"))

    messages.append(HumanMessage(content=question))

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    resp = llm.invoke(messages)
    return resp.content, docs

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

if "history" not in st.session_state:
    st.session_state.history = []

if "tmp_paths" not in st.session_state:
    st.session_state.tmp_paths = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.sidebar:
    st.header("Documents")
    current_key = f"uploader_{st.session_state.uploader_key}"
    uploaded = st.file_uploader("Please upload one or many .txt/.md/.pdf file(s)", accept_multiple_files=True, type=["txt", 'md', "pdf"], key=current_key,)
    if uploaded:
        if st.button("Index uploaded files"):
            # Upload the files
            print("Uploaded files:", [f.name for f in uploaded])
            with st.spinner("Your file(s) are being Indexed..."):
                tmp_paths = []
                for f in uploaded:
                    suffix = os.path.splitext(f.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.read())
                        tmp_paths.append(tmp.name)
                
                # Load, the documents then chunk it and then embed them in Vector Store
                docs = load_documents(tmp_paths)
                chunks = get_document_chunks(docs)
                build_or_update_index(chunks)
            st.success("Indexing completed successfully!!!")
    else:
        st.info("Upload files first.")

    stats = list_index_stats()
    st.caption(f"Vector store: {stats['collection']} — {stats['docs']} chunks")

    if st.button("Erase uploaded file(s)"):
        errors = []

        # Delete the Chroma collection via the LangChain wrapper's underlying client
        try:
            emb = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", EMBED_MODEL))
            vs = Chroma(
                collection_name=COLLECTION,
                embedding_function=emb,
                persist_directory=PERSIST_DIR,
            )
            vs._client.delete_collection(COLLECTION)
        except Exception as e:
            # If the collection doesn't exist or client is locked, fall back to nuking the directory
            errors.append(f"_client.delete_collection: {e}")
            try:
                if os.path.isdir(PERSIST_DIR):
                    shutil.rmtree(PERSIST_DIR)
            except Exception as e2:
                errors.append(f"rmtree fallback: {e2}")

        # Delete all temp files we created for uploads
        for p in list(st.session_state.get("tmp_paths", [])):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                errors.append(f"remove {p}: {e}")
        st.session_state["tmp_paths"] = []

        # Reset chat and uploader
        st.session_state["history"] = []

        # Clear the uploader's widget state and bump its key to visually reset it
        st.session_state.pop(current_key, None)   # drop any residual widget value
        st.session_state.uploader_key += 1        # force a brand-new uploader widget instance
        st.rerun()
        
        # Refresh the app to reflect cleared state
        st.info("Upload files first.")
        stats = list_index_stats()
        st.caption(f"Vector store: {stats['collection']} — {stats['docs']} chunks")

        # Feedback
        if errors:
            st.warning("Cleared with warnings:\n" + "\n".join(f"• {e}" for e in errors))
        else:
            st.success("Cleared: collection, temp uploads, and chat state. Ready to re-index.")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["assistant"])

if prompt := st.chat_input("Ask any questions about the file(s) you uploaded here…"):
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Reading file(s) and processing answer.."):
            answer, docs = answer_question(prompt, st.session_state.history)
            st.write(answer)
    st.session_state.history.append({"user": prompt, "assistant": answer})
