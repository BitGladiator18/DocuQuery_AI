import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="📎 MAANG LLM Assistant")
st.title("📘 Upload a PDF and Ask Questions")

uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")
query = st.text_input("🔍 Ask a question based on the PDF")

if uploaded_file and query:
    with st.spinner("Processing..."):

        # Save PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load and chunk PDF using Recursive splitter
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Vector Store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        docs = vectorstore.similarity_search(query, k=3)

        # Load T5 model using HuggingFace pipeline (Free-tier model)
        model_id = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
        llm = HuggingFacePipeline(pipeline=pipe)

        # QA Chain
        chain = load_qa_chain(llm, chain_type="stuff")
        result = chain.run(input_documents=docs, question=query)

        st.success(result)
