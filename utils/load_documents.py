from langchain_community.document_loaders import TextLoader
import os

def load_docs(folder_path):
    docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, file_name))
            docs.extend(loader.load())
    return docs

