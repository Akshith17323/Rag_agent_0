import streamlit as st
import os
from dotenv import load_dotenv

# Import components from our other modules
from rag import load_documents, build_vectorstore, retrieve_context
from web_search import search_web, format_search_results

# LangChain Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Try to load local .env if available
load_dotenv()

# --- Utility Functions ---
def get_api_key(key_name: str) -> str:
    """Safely fetch API keys: prioritize st.secrets (deployed), then os.getenv (local)."""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, "")


def init_session_state():
    """Initialize essential Streamlit session state variables."""
    if "vectorstore" not in st.session_state:
         st.session_state["vectorstore"] = None
    if "api_key_status" not in st.session_state:
         st.session_state["api_key_status"] = {"google": False, "serper": False}

# --- Main Streamlit App ---

st.set_page_config(page_title="RAG + Web Search Agent", page_icon="🕵️", layout="wide")

init_session_state()

# Fetch keys
google_key = get_api_key("GOOGLE_API_KEY")
serper_key = get_api_key("SERPER_API_KEY")

st.session_state["api_key_status"]["google"] = bool(google_key)
st.session_state["api_key_status"]["serper"] = bool(serper_key)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("RAG + Web Search Agent")

    # API Status Indicators
    st.markdown("### API Status")
    google_color = "🟢" if st.session_state["api_key_status"]["google"] else "🔴"
    st.markdown(f"**Google API Key**: {google_color}")
    
    serper_color = "🟢" if st.session_state["api_key_status"]["serper"] else "🔴"
    st.markdown(f"**Serper API Key**: {serper_color}")
    
    if not st.session_state["api_key_status"]["google"]:
        st.warning("GOOGLE_API_KEY is missing. RAG and answering will fail.")
        
    st.markdown("---")

    # Document Uploader Configuration
    st.markdown("### Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            try:
                with st.spinner("Processing and indexing documents..."):
                    # Extract bytes and names for rag.py 
                    file_bytes_list = []
                    file_names = []
                    
                    for uploaded_file in uploaded_files:
                        file_bytes_list.append(uploaded_file.getvalue())
                        file_names.append(uploaded_file.name)
                        
                    # Build RAG system
                    docs = load_documents(file_bytes_list, file_names)
                    if docs:
                        # Set GOOGLE_API_KEY in environment explicitly so langchain_google_genai finds it 
                        if google_key and not os.getenv("GOOGLE_API_KEY"):
                             os.environ["GOOGLE_API_KEY"] = google_key
                             
                        vstore = build_vectorstore(docs)
                        if vstore:
                             st.session_state["vectorstore"] = vstore
                             st.success(f"{len(uploaded_files)} document(s) loaded and indexed!")
                        else:
                             st.error("Failed to build vector store.")
                    else:
                        st.error("No valid documents could be loaded.")
            except Exception as e:
                st.error(f"Error processing documents: {e}")
        else:
             st.warning("Please upload a file first.")

    st.markdown("---")
    
    # Web search toggle
    enable_web_search = st.checkbox("Enable Live Web Search (Serper)", value=True)


# ==========================================
# MAIN AREA
# ==========================================
st.header("Ask a Question")

question = st.text_area("What would you like to know?", height=100)

if st.button("Get Answer", type="primary"):
    
    # Validate inputs
    has_rag = st.session_state["vectorstore"] is not None
    
    if not has_rag and not enable_web_search:
        st.warning("Please upload documents or enable web search to get an answer.")
        st.stop()
        
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not st.session_state["api_key_status"]["google"]:
        st.error("Google API Key is required to generate an answer.")
        st.stop()

    with st.spinner("Searching and generating answer..."):
        try:
            # 1. RAG Context Retrieval
            rag_context = ""
            if has_rag:
                rag_context = retrieve_context(st.session_state["vectorstore"], question)
                
            # 2. Web Search Retrieval
            web_results = []
            web_context = ""
            if enable_web_search and st.session_state["api_key_status"]["serper"]:
                # Ensure Serper key is in env for web_search.py
                if serper_key and not os.getenv("SERPER_API_KEY"):
                     os.environ["SERPER_API_KEY"] = serper_key

                web_results = search_web(question)
                web_context = format_search_results(web_results)
            elif enable_web_search and not st.session_state["api_key_status"]["serper"]:
                 st.warning("Web search enabled but Serper API Key missing.")

            # 3. Build Prompt for Gemini
            prompt_template = """System: You are a helpful research assistant. Answer the question using the provided context. If the combined context is insufficient, say so clearly.

User:
Document Context:
{rag_context}

Web Search Results:
{web_context}

Question: {question}

Provide a clear, accurate, and consolidated answer based ONLY on the context provided above.
"""
            prompt = PromptTemplate.from_template(prompt_template)
            
            # Ensure Google key is in OS env for ChatGoogleGenerativeAI
            if google_key and not os.getenv("GOOGLE_API_KEY"):
                 os.environ["GOOGLE_API_KEY"] = google_key

            # 4. Generate Answer with Gemini
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
            
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({
                "rag_context": rag_context if rag_context else "No document context available.",
                "web_context": web_context if web_context else "No web search context available.",
                "question": question
            })
            
            # 5. Display Answer
            st.success("Answer generated successfully!")
            st.markdown(response)
            
            st.markdown("---")
            st.subheader("Results Debugging")
            
            # RAG Debug Expander
            with st.expander("View retrieved document chunks"):
                if has_rag:
                     st.text(rag_context)
                else:
                     st.info("No documents indexed. RAG retrieval was skipped.")
                     
            # Web Search Debug Expander
            with st.expander("View web search results"):
                 if enable_web_search and web_results:
                     for idx, result in enumerate(web_results, 1):
                          st.markdown(f"**[{idx}] [{result.get('title')}]({result.get('link')})**")
                          st.caption(result.get('snippet'))
                 elif enable_web_search and not web_results:
                     st.warning("Web search ran but returned no results.")
                 else:
                     st.info("Web search was disabled.")

        except Exception as e:
            st.error(f"An error occurred while generating the answer: {e}")
