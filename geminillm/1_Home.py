import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from io import BytesIO

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Paths to the default PDFs
DEFAULT_PDFS = ["Constitution_Amendment.pdf", "constitution_of_india.pdf", "the_constitution.pdf"]

def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf))  # Use BytesIO to handle bytes objects
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle possible None values
    if not text:
        raise ValueError("No text could be extracted from the PDFs.")
    return text

def get_text_chunks(text):
    """Split the extracted text into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("Text splitting failed; no chunks were generated.")
    return chunks

def get_vector_store(chunks, store_name="faiss_index"):
    """Generate and save the FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore

    retries = 3
    delay = 5  # Start with a 5-second delay

    for attempt in range(retries):
        try:
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            vector_store.save_local(store_name)
            return vector_store
        except GoogleAPICallError as e:
            print(f"Google API Call Error: {e}")
            if "Rate limit" in str(e) or "quota exceeded" in str(e):
                print(f"Rate limit exceeded. Attempt {attempt + 1} of {retries}.")
                if attempt < retries - 1:
                    time.sleep(delay)  # Wait before retrying
                    delay *= 2  # Exponential backoff
                else:
                    raise e
            else:
                raise e
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise e

def get_conversational_chain():
    """Create a conversational chain for answering questions."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say,
    "I didn't find any relevant info in the pdf, then provide an answer from your side, search the internet for the answer in detail , and also
    provide the reference. Do not provide the wrong answer".\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

def user_input(user_question):
    """Handle user input and return a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore

    # Load the vector store
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        # Debugging: Log the retrieved documents for better insight
        print("Retrieved Documents:")
        for i, doc in enumerate(docs):
            print(f"Document {i + 1}: {doc.page_content[:200]}")  # Log the first 200 characters

        if not docs:
            st.error("No relevant context found in the indexed documents.")
            return "No relevant context found."
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print("Response Generated:", response)
        return response['output_text']
    except Exception as e:
        print(f"Error during similarity search or response generation: {e}")
        return "There was an error processing your request."

def main():
    st.set_page_config(
        page_title="Home",
        page_icon="logo.png"
    )

    st.sidebar.title("Welcome to Law Buddy")

    # Initialize default chunks
    default_chunks = []
    if not os.path.exists("faiss_index"):
        with st.spinner("Processing default PDFs..."):
            try:
                default_text = get_pdf_text([open(pdf, 'rb').read() for pdf in DEFAULT_PDFS])
                default_chunks = get_text_chunks(default_text)
                get_vector_store(default_chunks, store_name="faiss_index")
                st.success("Default PDFs processed successfully")
            except ValueError as e:
                st.error(f"Error processing default PDFs: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    else:
        st.info("Default FAISS index already exists. Skipping processing.")

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.write("### Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        # Process uploaded PDFs
                        raw_text = get_pdf_text([pdf.read() for pdf in pdf_docs])
                        text_chunks = get_text_chunks(raw_text)
                        # Combine default chunks with user-uploaded chunks
                        all_chunks = default_chunks + text_chunks
                        get_vector_store(all_chunks, store_name="faiss_index")
                        st.success("Done")
                    except ValueError as e:
                        st.error(f"Error processing uploaded PDFs: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
            else:
                st.warning("Please upload some PDFs to process.")

    # Main content area for displaying chat messages
    st.title("Chat with Law Buddy")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="bot.png"):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="😃"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="bot.png"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
