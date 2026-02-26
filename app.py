import os
try:
    import streamlit as st
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY or ""
os.environ["SERPER_API_KEY"] = SERPER_API_KEY or ""

# Import components from our other modules
from rag import load_documents, build_vectorstore, retrieve_context
from web_search import search_web, format_search_results

# LangChain Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def init_session_state():
    """Initialize essential Streamlit session state variables."""
    if "vectorstore" not in st.session_state:
         st.session_state["vectorstore"] = None
    if "api_key_status" not in st.session_state:
         st.session_state["api_key_status"] = {"google": False, "serper": False}
    if "chat_history" not in st.session_state:
         st.session_state["chat_history"] = []

# --- Main Streamlit App ---

st.set_page_config(page_title="RAG + Web Search Agent", page_icon="🌐", layout="wide")

# UI Enhancements (Custom CSS)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 30px;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-active { background-color: #d4edda; color: #155724; }
    .status-inactive { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

init_session_state()

st.session_state["api_key_status"]["google"] = bool(GOOGLE_API_KEY)
st.session_state["api_key_status"]["serper"] = bool(SERPER_API_KEY)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8637/8637113.png", width=60) # Placeholder AI Icon
    st.markdown("## ⚙️ Agent Configuration")
    st.divider()

    # API Status Indicators in a clean container
    st.markdown("#### 🔑 API Connections")
    
    col1, col2 = st.columns(2)
    with col1:
         if st.session_state["api_key_status"]["google"]:
             st.markdown('<span class="status-badge status-active">🟢 Google API</span>', unsafe_allow_html=True)
         else:
             st.markdown('<span class="status-badge status-inactive">🔴 Google API</span>', unsafe_allow_html=True)
    with col2:
         if st.session_state["api_key_status"]["serper"]:
             st.markdown('<span class="status-badge status-active">🟢 Serper API</span>', unsafe_allow_html=True)
         else:
             st.markdown('<span class="status-badge status-inactive">🔴 Serper API</span>', unsafe_allow_html=True)

    if not st.session_state["api_key_status"]["google"]:
        st.error("Missing Google API Key.")
        
    st.divider()

    # Web search toggle
    st.markdown("#### 🌍 Tools")
    enable_web_search = st.toggle("Enable Live Web Search", value=True, help="Uses the Serper API to pull live data from Google.")

    st.divider()

    # Document Uploader Configuration
    st.markdown("#### 📄 Knowledge Base (RAG)")
    uploaded_files = st.file_uploader(
        "Upload reference documents", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        help="Upload PDF or TXT files to inject local context into the agent."
    )
    
    if st.button("Index Documents", use_container_width=True, type="secondary"):
        if uploaded_files:
            try:
                with st.spinner("Processing & Vectorizing..."):
                    # Extract bytes and names for rag.py 
                    file_bytes_list = []
                    file_names = []
                    for uploaded_file in uploaded_files:
                        file_bytes_list.append(uploaded_file.getvalue())
                        file_names.append(uploaded_file.name)
                        
                    # Build RAG system
                    docs = load_documents(file_bytes_list, file_names)
                    if docs:
                        vstore = build_vectorstore(docs)
                        if vstore:
                             st.session_state["vectorstore"] = vstore
                             st.toast(f"Successfully indexed {len(uploaded_files)} document(s)!", icon="✅")
                        else:
                             st.error("Failed to build vector store.")
                    else:
                        st.error("No valid documents could be loaded.")
            except Exception as e:
                st.error(f"Error processing documents: {e}")
        else:
             st.toast("Please upload files before indexing.", icon="⚠️")

    if st.session_state["vectorstore"]:
         st.success("📚 Knowledge Base Active")


# ==========================================
# MAIN AREA
# ==========================================
st.markdown('<p class="main-header">Hybrid AI Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Retrieval-Augmented Generation & Live Web Search</p>', unsafe_allow_html=True)

# Display Chat History
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"], avatar="🕵️" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# Quick clear button
if st.session_state["chat_history"]:
     if st.button("Clear Conversation", type="tertiary"):
          st.session_state["chat_history"] = []
          st.rerun()

# Chat Input
if question := st.chat_input("Ask me anything..."):
    
    # 1. Show user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    
    st.session_state["chat_history"].append({"role": "user", "content": question})

    # Validate inputs seamlessly
    has_rag = st.session_state["vectorstore"] is not None
    
    if not has_rag and not enable_web_search:
        with st.chat_message("assistant", avatar="🕵️"):
            warning_msg = "⚠️ I need access to information! Please upload documents to the Knowledge Base or enable Live Web Search in the sidebar."
            st.warning(warning_msg)
            st.session_state["chat_history"].append({"role": "assistant", "content": warning_msg})
        st.stop()
        
    if not st.session_state["api_key_status"]["google"]:
        with st.chat_message("assistant", avatar="🕵️"):
            st.error("Google API Key is required to power my brain. Please configure it.")
        st.stop()

    # Process Answer
    with st.chat_message("assistant", avatar="🕵️"):
        status_text = st.empty()
        status_text.text("🤔 Thinking and gathering context...")
        
        try:
            # RAG Context Retrieval
            rag_context = ""
            if has_rag:
                status_text.text("📚 Scanning Knowledge Base...")
                rag_context = retrieve_context(st.session_state["vectorstore"], question)
                
            # Web Search Retrieval
            web_results = []
            web_context = ""
            if enable_web_search and st.session_state["api_key_status"]["serper"]:
                status_text.text("🌍 Searching the live web...")
                web_results = search_web(question)
                web_context = format_search_results(web_results)
            elif enable_web_search and not st.session_state["api_key_status"]["serper"]:
                 st.toast("Web search enabled but Serper API Key missing.", icon="⚠️")

            # Build Prompt for Gemini
            status_text.text("🧠 Synthesizing answer...")
            prompt_template = """System: You are an intelligent, helpful research assistant. Answer the user's question clearly and accurately using the provided context. If the combined context is insufficient to answer the question, say so clearly.

User Context Inputs:
[DOCUMENT KNOWLEDGE BASE STNIPPETS]:
{rag_context}

[LIVE WEB SEARCH RESULTS]:
{web_context}

Question: {question}

Provide a definitive, well-formatted markdown response based ONLY on the context provided above.
"""
            prompt = PromptTemplate.from_template(prompt_template)

            # Generate Answer with Gemini
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({
                "rag_context": rag_context if rag_context else "No document context available.",
                "web_context": web_context if web_context else "No web search context available.",
                "question": question
            })
            
            # Display final answer
            status_text.empty()
            st.markdown(response)
            
            # Save to history
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            
            # Context Debugging Section
            st.markdown("---")
            col_rag, col_web = st.columns(2)
            
            with col_rag:
                with st.expander("📚 RAG Context Used", expanded=False):
                    if has_rag:
                         st.caption(rag_context)
                    else:
                         st.info("System not utilized.")
                         
            with col_web:
                with st.expander("🌍 Web Search Results Used", expanded=False):
                     if enable_web_search and web_results:
                         for idx, result in enumerate(web_results, 1):
                              st.markdown(f"**[{idx}] [{result.get('title')}]({result.get('link')})**")
                              st.caption(result.get('snippet'))
                     else:
                         st.info("System not utilized or no results.")

        except Exception as e:
            status_text.empty()
            st.error(f"An error occurred while generating the answer: {e}")
