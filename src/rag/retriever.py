from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parents[2]
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# FAISS returns L2 distance: lower score = more similar.
# Documents with a distance above this threshold are considered irrelevant.
DEFAULT_SCORE_THRESHOLD = 1.15


class KnowledgeRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, k: int = 4):
        """
        Return only the documents that are semantically close enough
        to the query. If nothing passes the threshold, return an empty list.
        """
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k
        )

        relevant_documents = [
            document
            for document, score in results
            if score <= DEFAULT_SCORE_THRESHOLD
        ]

        return relevant_documents

    def retrieve_with_scores(self, query: str, k: int = 4):
        """
        Same as retrieve(), but returns (document, score) tuples.
        Useful for debugging and evaluation.
        """
        return self.vectorstore.similarity_search_with_score(
            query,
            k=k
        )