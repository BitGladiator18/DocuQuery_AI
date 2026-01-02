from langchain_community.vectorstores import FAISS

def create_vectorstore(documents, embedding):
    return FAISS.from_documents(documents, embedding)
