import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with API key
genai_api_key = os.getenv("GOOGLE_API_KEY")

# Paths to the default PDFs
DEFAULT_PDFS = ["Constitution_Amendment.pdf", "constitution_of_india.pdf"]

def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the extracted text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks, store_name="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    retries = 3
    delay = 5  # Start with a 5-second delay

    for attempt in range(retries):
        try:
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local(store_name)
            return vector_store
        except ResourceExhausted as e:
            print(f"Rate limit exceeded. Attempt {attempt + 1} of {retries}.")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
                delay *= 2  # Exponential backoff
            else:
                raise e

    if vector_store:
        vector_store.save_local(save_path)
        print("FAISS index saved successfully.")
    else:
        print("Failed to process and save FAISS index.")


def main():

    if os.path.exists("faiss_index.index"):
        print("Default PDFs processed")
        return
    # Extract text from default PDFs
    print("Extracting text from default PDFs...")
    default_text = get_pdf_text(DEFAULT_PDFS)

    # Split text into chunks
    print("Splitting text into chunks...")
    default_chunks = get_text_chunks(default_text)

    # Embed chunks and save them using FAISS
    print("Embedding text and saving FAISS index...")
    get_vector_store(default_chunks)

    print("Default PDFs processed and FAISS index saved successfully.")

if __name__ == "__main__":
    main()
