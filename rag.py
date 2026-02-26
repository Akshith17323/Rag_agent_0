import os
import tempfile
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

def load_documents(file_bytes_list: list, file_names: list) -> list[Document]:
    """
    Load documents from raw bytes and file names into LangChain Document objects.
    
    Args:
        file_bytes_list (list): List of raw file bytes from Streamlit uploader.
        file_names (list): List of corresponding file names.
        
    Returns:
        list[Document]: A list of LangChain Document objects.
    """
    documents = []
    
    for file_bytes, file_name in zip(file_bytes_list, file_names):
        # Handle PDF files
        if file_name.lower().endswith(".pdf"):
            # Create a temporary file to save the PDF bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
                
            try:
                # Use PyPDFLoader to load the temporary file
                loader = PyPDFLoader(tmp_file_path)
                pdf_docs = loader.load()
                # Optional: Add the original filename to metadata
                for doc in pdf_docs:
                    doc.metadata["source"] = file_name
                documents.extend(pdf_docs)
            finally:
                # Ensure the temporary file is deleted even if loading fails
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                    
        # Handle Text files
        elif file_name.lower().endswith(".txt"):
            try:
                # Decode bytes to string
                text_content = file_bytes.decode("utf-8")
                # Wrap in a LangChain Document
                doc = Document(
                    page_content=text_content,
                    metadata={"source": file_name}
                )
                documents.append(doc)
            except UnicodeDecodeError as e:
                print(f"Error decoding text file {file_name}: {e}")
                
        else:
            print(f"Unsupported file type for {file_name}. Skipping.")

    return documents


def build_vectorstore(documents: list[Document]):
    """
    Split documents, create embeddings, and store them in an in-memory Chroma vector database.
    
    Args:
        documents (list[Document]): The loaded LangChain Document objects.
        
    Returns:
        Chroma: An in-memory Chroma vectorstore instance. Returns None if no documents provided or error occurs.
    """
    if not documents:
        print("No documents provided to build vectorstore.")
        return None
        
    # 1. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        print("Document splitting resulted in no chunks.")
        return None

    # 2. Create embeddings using Google Gemini
    # Requires GOOGLE_API_KEY in environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment variables.")
        
    # Note: Ensure the model specified is correct and supported in your region
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 3. Store in Chroma (in-memory, no persist_directory)
    try:
         vectorstore = Chroma.from_documents(
             documents=chunks,
             embedding=embeddings
         )
         return vectorstore
    except Exception as e:
        print(f"Error building vectorstore: {e}")
        return None

def retrieve_context(vectorstore, query: str) -> str:
    """
    Retrieve relevant context from the vector store for a given query.
    
    Args:
        vectorstore (Chroma): The Chroma vectorstore instance.
        query (str): The search query.
        
    Returns:
        str: A formatted string of retrieved chunks separated by newlines, 
             or a fallback message if no context found.
    """
    if not vectorstore:
         return "No relevant context found in documents."
         
    try:
        # Retrieve top 3 most relevant chunks
        results = vectorstore.similarity_search(query, k=3)
        
        if not results:
             return "No relevant context found in documents."
             
        # Extract the page content from the Document objects
        context_chunks = [doc.page_content.strip() for doc in results]
        
        # Join the chunks with the specified separator
        joined_context = "\n---\n".join(context_chunks)
        return joined_context
        
    except Exception as e:
        print(f"Error during context retrieval: {e}")
        return "No relevant context found in documents."

if __name__ == "__main__":
    # Optional test block
    print("rag.py module loaded successfully.")
