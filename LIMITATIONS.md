# Research Assistant — Known Limitations

This document lists all known limitations of the current system as of the time of writing.
These are not failures — they are honest engineering tradeoffs made to ship a working product
and then improve it incrementally. Every limitation listed here has a known fix path.

---

## Data Limitations

**Scanned PDFs not readable**
PyPDFLoader can only extract text that is embedded in the PDF file.
Scanned documents are essentially images — there is no text to extract.
The system will load 0 pages or garbage characters from scanned files.
Fix: add OCR using pytesseract before loading.

**Tables and figures ignored**
Charts, graphs, chemical structures, western blots, and all visual content
are completely invisible to the system. It only reads text.
For scientific papers this is significant — many key findings are in figures.
Fix: multimodal models (GPT-4o, LLaVA) that can read images. Planned as version 2.0.

**Two-column PDF layouts**
Research papers often use two-column formatting.
PyPDFLoader reads left-to-right across the full page width,
mixing content from both columns into gibberish.
Fix: pdfplumber handles column layouts better than PyPDFLoader.

**Password-protected or corrupted PDFs**
The system fails silently on these — it loads 0 pages with no clear error message.
Fix: add validation before processing to detect and warn the user.

---

## Answer Quality Limitations

**Small local model (llama3.2)**
The 2GB llama3.2 model running on CPU is capable but limited compared to
larger cloud models like GPT-4o. It sometimes ignores instructions, gives
inconsistent answers to the same question, and is slower than cloud APIs.
Fix: switch to OpenAI API or Claude API for better answer quality.

**Conversation memory unreliable**
The agent does not reliably remember previous answers in the same conversation.
Follow-up questions that reference earlier answers often fail.
Root cause: llama3.2 is too small to maintain context reliably across many turns.
Fix: larger model, or a vector-based memory system that retrieves relevant history.

**Repeated chunks in retrieval**
ChromaDB sometimes returns the same chunk multiple times from the same paper.
This wastes context window space and can skew answers toward one source.
Fix: deduplicate chunks by content before passing to the LLM.

**Bracketed citation numbers in answers**
Numbers like [39] or [11] from the original papers sometimes appear in answers.
These are meaningless to users who do not have the original paper open.
Fix: stronger post-processing to strip all bracketed numbers from responses.

**k=4 chunks may miss relevant content**
We retrieve only the top 4 most similar chunks per question.
For complex multi-part questions this may not be enough context.
Fix: increase k, or implement re-ranking to get better quality chunks.

**No re-ranking**
We use raw cosine similarity to find chunks. The top 4 by similarity
are not always the 4 most useful for answering the question.
Fix: add a re-ranker model (cross-encoder) to re-order chunks by relevance.

---

## Scale Limitations

**One user at a time locally**
Ollama processes one request at a time on your CPU.
Multiple simultaneous users cause queuing and slow responses.
Fix: cloud hosting with GPU, or OpenAI API which handles concurrency automatically.

**Large document collections are slow**
Embedding is done on CPU using Ollama. Processing 1000 documents
per session would take hours and strain local memory.
Fix: cloud embedding API (OpenAI text-embedding-3-small), cloud vector database.

**Local ChromaDB struggles at scale**
ChromaDB running locally works well up to roughly 50,000 chunks.
Beyond that, search slows down and memory becomes a constraint.
Fix: Pinecone, Weaviate, or pgvector in the cloud for production scale.

**Session-based storage only**
User-uploaded documents exist only for the duration of their browser session.
When they close the browser everything is gone.
There are no persistent user accounts or saved document collections.
Fix: user authentication system, persistent cloud storage per user.

---

## Security Limitations

**No user authentication**
Anyone with the public URL can use the app.
There is no login, no user accounts, and no access control.
Fix: add authentication (Streamlit built-in auth, Auth0, or similar).

**No prompt injection protection**
A malicious user could craft inputs designed to hijack the system prompt
or extract information they should not have access to.
Current protection is limited to the system prompt wording.
Fix: add an input validation layer before passing to the LLM.

**API key management**
API keys are stored in a .env file locally which works for development.
Cloud deployment requires proper secrets management.
Fix: use Streamlit Cloud secrets management or environment variables in deployment.

---

## Product Limitations

**English only**
The system works best with English language documents and questions.
Non-English papers will load but answer quality will be significantly worse.
Fix: multilingual embedding model, multilingual LLM.

**No conversation export**
Users cannot save, download, or export their conversation history.
There is no PDF report generation from Q&A sessions.
Fix: add an export button that generates a formatted PDF of the conversation.

**No user feedback mechanism**
Users cannot flag wrong answers or rate responses.
Without feedback there is no systematic way to identify and fix quality issues.
Fix: thumbs up/down rating per answer, logged to a database for review.

**No answer citations linking to exact paper location**
Answers show the source paper filename but not the exact page or section.
Users have to search the paper manually to verify the answer.
Fix: store page numbers in chunk metadata and display them with answers.

**No incremental upload within a session**
If a user uploads additional documents after already asking questions,
the entire document collection is re-processed from scratch and
conversation history is lost. Each file addition triggers a full restart.
Fix: track already-processed files in session_state and only embed new additions.

---

## The Deployment Blocker — Most Important

**Ollama cannot run on Streamlit Cloud**
Streamlit Cloud is a lightweight hosting service that cannot run large local models.
Our entire pipeline depends on Ollama for both embeddings (nomic-embed-text)
and answer generation (llama3.2).

To deploy publicly the following must change:
- Switch embeddings from OllamaEmbedder to OpenAI text-embedding-3-small
- Switch LLM from llama3.2 to gpt-4o-mini or claude-haiku
- Store API keys as Streamlit Cloud secrets instead of .env file
- Re-embed all documents using the new embedding model
- Update all three scripts (embed_and_store.py, ask_question.py, agent.py) and app.py

This is the next major engineering task before a true public deployment.
A workaround is to deploy on a cloud server (Render, Railway, AWS EC2)
that can run Ollama — more complex setup but keeps the local model approach.

---

## What Is Working Well

Despite these limitations the system successfully:
- Loads and processes any text-based PDF document
- Creates searchable vector embeddings of document content
- Answers questions grounded in uploaded documents with source citations
- Falls back to web search when documents cannot answer
- Handles basic math calculations
- Provides a clean browser-based chat interface
- Tracks and indexes new documents incrementally
- Passed evaluation at 0.79 faithfulness score

These limitations are the roadmap for version 2.0.

---

## Monitoring Limitations

**File-based logging does not persist on cloud deployment**
Currently the app logs every question, tool used, and answer to a local logs.txt file.
This works perfectly for local development and testing.
However when deployed to Streamlit Cloud the file system resets on every redeploy
meaning all logs are wiped. The logs.txt approach only works when the app runs locally.
Fix: store logs in a cloud database (Supabase free tier works well) or use LangSmith
which is a hosted monitoring service built specifically for LLM applications.
LangSmith integration is planned for version 2.0 when the app has real production traffic.

**No log rotation**
The logs.txt file grows indefinitely. Every question adds new lines.
For local testing this is not a problem — even millions of queries produce a small file.
For production this needs log rotation — splitting into daily files and 
deleting files older than 30 days automatically.
Fix: implement Python's built-in logging module with RotatingFileHandler.

*Last updated: 08/24/2026.*
