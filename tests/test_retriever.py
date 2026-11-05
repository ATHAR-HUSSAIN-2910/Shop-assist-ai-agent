from src.rag.retriever import KnowledgeRetriever

retriever = KnowledgeRetriever()

query = "Can I return a product after 20 days?"

documents = retriever.retrieve(query, k=2)

for document in documents:
    print("SOURCE:", document.metadata.get("source"))
    print("CONTENT:")
    print(document.page_content)
    print("-" * 60)