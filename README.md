# 🧠 RAG Chatbot with Long & Short Term Memory

An intelligent conversational AI that combines **Retrieval-Augmented Generation (RAG)** with **persistent memory** — answering questions from your PDF documents while remembering who you are across conversations.

---

## 🚀 What It Does

Most chatbots forget everything the moment you close them. This one doesn't.

- Ask questions about any PDF → answers come from the document
- It remembers important things you share (your name, preferences, goals)
- It summarizes long conversations automatically to stay efficient
- Built with **LangGraph** for full control over the agent's reasoning flow

**Example:**
> "My name is Ali and I'm studying machine learning" → remembered forever  
> "What does chapter 3 say about neural networks?" → retrieved from your PDF  
> Come back next week → it still remembers you're Ali

---

## ✨ Features

- 📄 **PDF RAG** — semantic search over any uploaded document
- 🧠 **Long-Term Memory** — extracts and stores important user information persistently
- 💬 **Short-Term Memory** — summarizes long conversations to avoid context overflow
- 🔁 **LangGraph Pipeline** — structured node-based agent flow
- 💾 **SQLite Checkpointing** — conversation history saved to local database
- ⚡ **Fast Inference** — powered by Groq (llama-3.1-8b-instant)
- 🔍 **Threshold-based Retrieval** — only retrieves context when similarity score is high enough

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace (BAAI/bge-base-en-v1.5) |
| Vector Store | ChromaDB |
| PDF Loader | LangChain PyPDFLoader |
| Memory (Short-Term) | LangGraph SqliteSaver |
| Memory (Long-Term) | LLM Extraction → State |
| Language | Python 3.11+ |

---

## ⚙️ How It Works

```
User Message
      ↓
┌─────────────────┐
│  Memory Node    │  → Extracts long-term info from user message
└────────┬────────┘
         ↓
┌─────────────────┐
│  Summary Node   │  → Summarizes past conversation if too long (6+ messages)
└────────┬────────┘
         ↓
┌─────────────────┐
│   RAG Node      │  → Searches PDF vector store for relevant context
└────────┬────────┘
         ↓
┌─────────────────┐
│   Chat Node     │  → LLM generates final answer using context + memory
└────────┬────────┘
         ↓
      Response
```

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/alirazaaihub/rag-memory-chatbot.git
cd rag-memory-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Open `agent.py` and replace the API key:

```python
GROQ_API_KEY = "your_groq_api_key_here"
```

Get your free Groq API key at: https://console.groq.com

---

## ▶️ Usage

**Step 1 — Build the vector store from your PDF:**

```bash
python ingest.py
```

> Edit `ingest.py` and set your PDF path:
> ```python
> file_path = "your_document.pdf"
> ```

**Step 2 — Start the chatbot:**

```bash
python agent.py
```

**Step 3 — Chat:**

```
Chatbot ready (type 'exit' to quit)

You: My name is Ali and I want to learn machine learning
AI: Nice to meet you Ali! I'll remember that...

You: What does the book say about embeddings?
AI: Based on the document, embeddings are...

You: exit
```

---

## 📁 Project Structure

```
rag-memory-chatbot/
│
├── ingest.py          # PDF loading, chunking, vector store creation
├── agent.py           # LangGraph agent with memory nodes
├── requirements.txt   # Project dependencies
├── vector_store/      # ChromaDB vector store (auto-created)
├── chat_memory.db     # SQLite conversation history (auto-created)
└── README.md
```

---

## 📋 Dependencies

```
langchain>=1.2.15
langchain-chroma>=1.1.0
langchain-community>=0.4.1
langchain-core>=1.2.26
langchain-groq>=1.1.2
langchain-huggingface>=1.2.1
langgraph>=1.1.6
sentence-transformers>=5.3.0
```

Install all at once:
```bash
pip install langchain langchain-chroma langchain-community langchain-groq langchain-huggingface langgraph sentence-transformers chromadb pypdf
```

---

## 🧩 Key Concepts Used

| Concept | What it does in this project |
|--------|-------------------------------|
| **RAG** | Retrieves relevant PDF chunks before generating answers |
| **LangGraph StateGraph** | Controls the flow between memory, RAG, and chat nodes |
| **Long-Term Memory** | LLM extracts user info and stores it in agent state |
| **Short-Term Memory** | Conversation summarized when messages exceed 6 turns |
| **SqliteSaver** | Persists conversation history across sessions by thread_id |
| **Similarity Score Threshold** | Only retrieves context if relevance score > 0.3 |
| **HuggingFace Embeddings** | Converts text to vectors using BAAI/bge-base-en-v1.5 |

---

## 🙋 About

Built by **Ali raza** — an 18-year-old self-taught AI developer from Pakistan.  
This is part of my Agentic AI portfolio built using LangChain, LangGraph, and Groq.

📌 [LinkedIn](https://www.linkedin.com/in/ali-raza-7124a0403/)

---

## 📄 License

MIT License — feel free to use and modify.
