from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def build_vectorstore():

    print("Loading knowledge base...")

    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating FAISS vector store...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTORSTORE_DIR.mkdir(exist_ok=True)

    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"Vector store saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()