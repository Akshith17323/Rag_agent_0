# 🕵️ RAG + Web Search AI Agent

This project is a hybrid Artificial Intelligence agent that combines **Retrieval-Augmented Generation (RAG)** with **Live Web Search** to answer complex queries. It uses Google's Gemini models to synthesize information from uploaded documents (PDFs, TXT) and real-time internet data to provide highly accurate, grounded answers.

---

## 🎓 My Learning Journey & "Vibe Coding"

I built this project to deeply understand how modern AI systems, RAG architectures, and tool-calling agents work under the hood.

Rather than submitting a standard college assignment or broadly auto-generating a boilerplate app, I built this using an AI-assisted **"vibe coding"** approach. This means I actively collaborated with AI as a pair-programming partner. I guided the architecture, made architectural decisions on how to handle API constraints, and learned the underlying mechanics of the code by iteratively solving real problems (like managing context windows, handling API `RESOURCE_EXHAUSTED` quotas, and routing logic).

Through this hands-on process, I learned and implemented:

- **Vector Databases & Embeddings**: Orchestrating LangChain document loaders (`PyPDFLoader`), Text Splitters, and in-memory Vector Stores (`ChromaDB`) using Google's embedding models.
- **Prompt Engineering**: Constructing robust system prompts that intelligently merge RAG context and live web search summaries.
- **API Integrations**: Handling external APIs (Google Gemini, Serper Web Search) securely and efficiently with timeouts and error fallbacks.
- **Deployment Mechanics**: Managing environmental variables safely between local `.env` setups and Streamlit Cloud secrets `.toml` formats.

This repository represents my practical, applied understanding of AI Engineering, forged through active building and troubleshooting.

---

## 🚀 Key Features

- **Document Ingestion**: Upload multiple PDFs and Text files. The app chunks, embeds, and indexes them locally.
- **Live Web Search**: Integrates with the Serper API to pull real-time data from Google Search for questions requiring up-to-date knowledge (e.g., events in 2025).
- **Context-Aware Synthesis**: Uses `gemini-2.5-flash-lite` to dynamically read both the retrieved document chunks and the web search snippets to provide a definitive, unified answer.
- **Debug Transparency**: Dedicated expander panels at the bottom of the app show exactly which document chunks and web links the AI used to generate its response.

## 🛠️ Setup Instructions

1. **Clone the repository** and navigate to the folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **API Keys**:
   - Create a `.env` file in the root directory.
   - Add your Google Gemini and Serper API keys:
     ```env
     GOOGLE_API_KEY="your_google_key"
     SERPER_API_KEY="your_serper_key"
     ```
4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## 🧪 Testing the Agent

The `sample_docs/` folder contains test files to evaluate the RAG pipeline. Try asking questions specific to those texts, then turn on Live Web Search to ask about current events, or combine both to observe how the AI merges local data with live internet data!
