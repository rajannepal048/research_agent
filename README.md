# Research Assistant Application based on RAG (Retrieval Augmented Generation) and AI Agents

This project is based on building a product with universal research literature assistant
especially for the research student in undergraduate and graduate degrees who reads lots of 
research paper in their field for a literature review. The app user can upload any collection of PDF documents
and ask questions in plain English. Answers are grounded in your uploaded papers with source citations. 
When documents cannot answer, the system searches the web as a fallback.

Being that said, it is not only limited to the research students. The user from diverse background 
(history, finance, legal contracts) could upload their PDF document and ask questions to the systems. The system will
answer the user's question based on the information available in the uploaded document. 

I built this as a portfolio project to demonstrate end-to-end LLM application development
covering RAG pipelines, AI agents, evaluation, and deployment.

---

## What this application does?

- Upload any PDF documents such as research papers, reports, textbooks, anything text-based.
- Ask questions and get answers with source citations.
- Documents are always searched first. Web search is only as fallback if there are not enough 
information in uploaded papers to provide answers.
- Evaluation of this app is scored at **0.79 faithfulness** by using Anthropic Claude as the judge.
- Query logging for monitoring.

---

## Tech stack

| Component | Tool |
|---|---|
| Language model | Ollama + llama3.2 (local, free, private) |
| Embeddings | nomic-embed-text via Ollama |
| Vector database | ChromaDB |
| Orchestration | Custom Python (bypassed LangChain agents) |
| Web search | Tavily API |
| Evaluation | Anthropic Claude API |
| UI | Streamlit |

---

## How to run locally in your machine?

**1. Install Ollama and download models**
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**2. Clone the repository**
```bash
git clone https://github.com/rajannepal048/research_agent.git
cd research_agent
```

**3. Create virtual environment and install dependencies**
```bash
python -m venv venv
.\venv\Scripts\Activate   # Windows
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the project root:

- TAVILY_API_KEY = your_tavily_key
- ANTHROPIC_API_KEY = your_anthropic_key


**5. Run the app**
```bash
streamlit run app.py
```

---

## How to use this application?

1. Open the app in your browser at `localhost:8501`
2. Upload your PDF files using the sidebar
3. Wait for processing. The system chunks and embeds your documents
4. Ask questions in the chat input at the bottom
5. Answers appear with source citations

---

## Project Structure

```
research_agent/
├── app.py — Streamlit UI (Phase 3)
├── agent.py — AI agent with 3 tools (Phase 2)
├── ask_question.py — Terminal Q&A interface (Phase 1)
├── embed_and_store.py — Document embedding pipeline (Phase 1)
├── load_and_chunk.py — Learning script, shows chunking in isolation
├── evaluate.py — RAGAS-style evaluation using Claude
├── requirements.txt — Python dependencies
├── LIMITATIONS.md — Known limitations and fix roadmap
├── pdfs/ — Local document collection (not in GitHub)
├── research_db/ — ChromaDB database (not in GitHub)
└── .env — API keys (not in GitHub)
```


---

## Evaluation results

Evaluated using a custom evaluation framework with Anthropic Claude as the judge.

| Metric | Score |
|---|---|
| Faithfulness | 0.79 |
| Answer Relevancy | 0.69 |
| Result | PASS |

---

## Known Limitations

Please refer to [LIMITATIONS.md](LIMITATIONS.md) for a full list of known limitations
and planned fixes for next version or an incremental progress of the application.

---

## What is next?

- Switch from Ollama to OpenAI API for cloud deployment (TBD)
- Add user authentication (TBD)
- Add persistent document storage per user (TBD)
- Add LangSmith monitoring for production traffic (TBD)
- Multimodal support for figures and tables in papers (TBD)

---

*Built by Rajan Nepal - CS Masters, North Dakota State University*
*Portfolio Project demonstrating end-to-end LLM application development*

*Last Updated - 09/04/2026*