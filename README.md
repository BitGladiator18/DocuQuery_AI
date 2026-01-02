# DocuQuery AI: Intelligent Document Assistant 🤖

DocuQuery AI is a RAG (Retrieval-Augmented Generation) system that allows users to chat with their documents (PDFs, DOCX, Text) using LLMs. It uses vector embeddings to retrieve relevant context and provide precise answers.

## 🚀 Key Features
* **Multi-Format Support**: Query resumes, reports, and text files.
* **Vector Search**: Uses high-performance embeddings for semantic search.
* **Context-Aware**: The AI only answers based on the provided document data.

## 🛠️ Tech Stack
* **Language**: Python 3.x
* **AI Framework**: LangChain / LlamaIndex
* **Interface**: Streamlit
* **Vector DB**: FAISS / ChromaDB

## 📦 Installation & Setup
1. Clone the repo: `git clone https://github.com/BitGladiator18/DocuQuery_AI.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your `.env` file with your API keys.
4. Run the app: `streamlit run app/main.py`

## 📂 Project Structure
* `app/`: Main application logic and UI.
* `utils/`: Helper scripts for document loading and embedding.
* `data/`: Storage for documents to be indexed.
* `scripts/`: Automation for creating vector stores.