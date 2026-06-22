import os
import shutil
from dotenv import load_dotenv

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()  # loads .env file

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

faiss_index_path = "faiss_index"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# -------------------------
# 1. PROCESS FILE FUNCTION
# -------------------------
def process_file(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".txt":
        loader = TextLoader(file_path)
    elif extension == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Only .txt and .pdf files are supported.")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    if os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_path)


# -------------------------
# 2. ASK QUESTION FUNCTION
# -------------------------
def ask_question(query):

    if not os.path.exists(faiss_index_path):
        return "No document indexed yet."

    vectorstore = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])

    if not context.strip():
        return "No relevant context found."

    # ✅ FIXED: correct env usage
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Gemini API key missing. Check .env file."

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Context:\n{context}\n\nQuestion: {query}"
    )

    return response.text