from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load documents from data folder (txt, pdf, docx)
documents = []

txt_loader = DirectoryLoader("./data", glob="**/*.txt", loader_cls=TextLoader)
documents.extend(txt_loader.load())

pdf_loader = DirectoryLoader("./data", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents.extend(pdf_loader.load())

docx_loader = DirectoryLoader("./data", glob="**/*.docx", loader_cls=Docx2txtLoader)
documents.extend(docx_loader.load())

print(f"✅ Loaded {len(documents)} documents")

# 2. Split documents into smaller chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)
print(f"✂️ Split into {len(split_docs)} chunks")

# 3. Create embeddings using HuggingFace model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create FAISS vectorstore from embeddings and save it locally
vectorstore = FAISS.from_documents(split_docs, embedding)
vectorstore.save_local("vectorstore")
print("✅ Vector store saved to 'vectorstore'")
