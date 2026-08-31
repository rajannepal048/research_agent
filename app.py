# -------------------------PHASE 3 | STREAMLIT UI-------------------------
# What Phase 3 does?
#
# This is a Streamlit UI file. This is what turns our terminal scripts into a proper web app that anyone can use in the browser.
# Right now everything runs in the terminal. Only we can see it and only we can use it. Phase 3 wraps the same code in a proper 
# web interface so anyone can use it from a browser.
# 
# Phase 1+2 (what we built)
# -----> Terminal only
# -----> Only us, only on our machine
# -----> Not shareable
#
# Phase 3 (What we are building)
# It wraps everything that we built in Phase 1 and Phase 2 into a web interface. When someone visits our app URL, they will see:
# -----> Browser chat interface
# -----> Anyone with the URL can use it
# -----> File Uploader so users bring their own documents
# -----> The conversation displayed like a chat app
# -----> User can upload their own PDFs docs and get grounded answers with source/citations, all from a browser
# -----> Deployed publicly on Streamlit Cloud
# 
# Behind the scenes it calls the same code from "embed_and_store.py" and "agent.py", nothing changes in the logic. 
# Streamlit just puts a face on it.
#
# Why Streamlit and not something else?
# Streamlit     → Python only, no HTML/CSS/JavaScript needed
#                 20 lines of code = full chat interface
#                 Free deployment on Streamlit Cloud
#                 Industry standard for AI demos and prototypes
# Flask/FastAPI → more powerful but requires frontend knowledge
#                 too complex for what we need right now
# React         → full frontend framework, overkill for a demo
#
# THIS IS A PRODUCT. NOT A TERMINAL SCRIPT. A REAL WEB APP
# RUN WITH : "STREAMLIT RUN APP.PY"
# DO NOT RUN WITH: "PYTHON APP.PY"
#
# -------------------------BLOCK 1 | IMPORTS AND PAGE SETUP-------------------------

import os                              # access environment variables and file system
import streamlit as st                 # the web framework that creates the UI
import ollama                          # direct ollama library for embeddings and LLM
from dotenv import load_dotenv         # reads ".env" file to get API keys safely
from langchain_chroma import Chroma    # loads our ChromaDB vector database
from tavily import TavilyClient        # Tavily web search for agent fallback
import tempfile                        # creates temporary folders for an uploaded files
from pathlib import Path               # handles file paths cleanly across operating system

# load API keys from ".env" file
load_dotenv()

# configure the Streamlit page - This must be the first Streamlit command

st.set_page_config(
    page_title = "Research Assistant",     # browser tab title
    page_icon = "R",                       # browser tab icon
    layout = "wide"                        # use full width of the browser
)
# After running this block of code only, Tab title      → says "Streamlit"
# after Block 2 it will say "Research Assistant"                  
# Page           → blank right now
#                  each block we add will add visible elements
#
#
# -------------------------BLOCK 2 | PAGE TITLE, LOAD THE DATABASE AND AGENT FUNCTION-------------------------
#
# Display the app title and subtitle on the page
# Load the database and tools once using Streamlit's cache system
# st.chace_resource means this only runs once and not on every user interaction
#
# Page title and description visible to others

st.title("Research Assistant")
st.markdown("Upload your research documents and ask questions. Answers are grounded in your uploaded papers.")
st.divider()       # horizontal line to separate header from the main content

# custom embedding class. This will be identical to all other scripts

class OllamaEmbedder:
    def embed_documents(self, texts):    # embed a list of texts
        return [ollama.embeddings(model = "nomic-embed-text", prompt = t)["embedding"] for t in texts]
    def embed_query(self, text):         # embed a single query
        return ollama.embeddings(model = "nomic-embed-text", prompt = text)["embedding"]

@st.cache_resource        # load once, reuse across all interactions - makes app faster
def load_database():
    # loads the existing ChromaDB database from disk
    embeddings = OllamaEmbedder()
    vectorstore = Chroma(
        persist_directory = "research_db",   # folder where vectors are stored
        embedding_function = embeddings      # must match model used when building database
    )
    return vectorstore

@st.cache_resource # load once, reuse across all interactions
def load_tavily():
    # loads the Tavily web search client
    return TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))

# loads database and tools

vectorstore = load_database()
tavily_client = load_tavily()

# show database status in sidebar

# st.sidebar.title("System Status") # this line shows unnecessary clutter in the webpage
# st.sidebar.success(f"Database loaded: {vectorstore._collection.count()} chunks") # this line shows unnecessary clutter in the webpage
#
# Two things to address after running Block 2
# What Block 2 built? What we can see:
# -----> Left sidebar   → "System Status" with green "Database loaded: 4050 chunks"
# -----> Main area      → "Research Assistant" title and subtitle
# -----> Divider line   → separates header from content below
# Block 2 is working perfectly.

# At this stage it shows 4050 chunks to users on left bar:
# That is wrong for the final product. Regular users should not see our internal database statistics. That is developer information,
# not user information. The reason it is there right now is because we are still in development and it is useful for us to confirm 
# the database loaded correctly. Think of it as a debug display.
# For the final product the sidebar will show user-relevant information:
# Developer view (now)         User view (final)
# ────────────────────         ─────────────────
# Database: 4050 chunks   →    Your uploaded documents: 3 files
#                              Questions asked: 12
# We will update this when we build the file uploader in Block 3. The sidebar will then show how many documents the user uploaded,
# not our internal chunk count. For now it is fine as a development indicator. We can clean it up in the next block.
#
#
# -------------------------BLOCK 3 | FILE UPLOADER (USER UPLOADS THEIR OWN PDFs)-------------------------
# 
# Here we will add the file uploader to th sidebar. This is the core feature that makes the app universal and any user can upload their own
# PDFs document here, the system processes them on the fly (chunks, embeds, and stores) in a temporary database specific to this session,
# and they can ask any questions about their own documents. When files are uploaded we run the same embedding
# logic from "embed_an_store.py" but dynamically for the uploaded files.
# We will also replace the developer chunk count with user friendly information.
# This is what makes the app universal and not locked to our pre-loaded papers.

from langchain_community.document_loaders import PyPDFLoader         # reads the PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits into chunks

def process_uploaded_files(uploaded_files):
    # takes uploaded PDF files, chunks them, embeds them, returns a vectorstore
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,     # same settings as "embed_and_store.py"
        chunk_overlap = 50    # consistency across the whole pipeline
    )

    for uploaded_file in uploaded_files:
        # save uploaded file temporarily so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())     # getvalue() reads all bytes correctly in Streamlit
            tmp_path = tmp.name                     # save the temp file path
        
        # load and chunk the PDF
        loader = PyPDFLoader(tmp_path)          # read the temp PDF file
        docs = loader.load()                    # extract all pages as text

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name    # tag with original filename

        chunks = splitter.split_documents(docs)            # split into chunks
        all_chunks.extend(chunks)                          # add to the main list 

        # clean up temp file after preprocessing
        os.unlink(tmp_path)                                # delete temp file. We will no longer need it.

    # embed chunks manually using our custom embedder

    # then pass directly to ChromaDB — bypasses the LangChain wrapper issue
    embeddings = OllamaEmbedder()
    texts = [chunk.page_content for chunk in all_chunks]          # extract text from chunks
    texts = [chunk.page_content for chunk in all_chunks]          # extract text from chunks
    metadatas = [chunk.metadata for chunk in all_chunks]          # extract metadata
    
    # embed all texts using nomic-embed-text
    embedded_vectors = embeddings.embed_documents(texts)  # get vectors for all chunks
    
    # create ChromaDB collection directly with pre-computed embeddings
    import chromadb
    chroma_client = chromadb.Client()  # in-memory client, no disk storage
    collection = chroma_client.get_or_create_collection("user_docs")  # get or create collection
    
    # add to collection with IDs
    ids = [str(i) for i in range(len(texts))]  # generate simple IDs
    collection.add(
        embeddings=embedded_vectors,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    # wrap in LangChain Chroma for compatibility with our search functions
    new_vectorstore = Chroma(
        client=chroma_client,
        collection_name="user_docs",
        embedding_function=embeddings
    )
    
    return new_vectorstore, len(all_chunks)

# -------------------------SIDEBAR-------------------------

st.sidebar.title("Your Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Files",               # label shown above the upload button
    type = "pdf",                     # only accept PDF files
    accept_multiple_files = True      # allow uploading multiple files at once
)

# handle uploaded files

if uploaded_files:
    with st.sidebar:
        with st.spinner("Processing your documents..."):     # loading indicator
            user_vectorstore, chunk_count = process_uploaded_files(uploaded_files)
        st.success(f"Ready! {len(uploaded_files)} files, {chunk_count} chunks indexed.")
    
    # use the user's vectorstore for questions
    active_vectorstore = user_vectorstore
else:
    # no files uploaded. Tell user to upload their documents
    active_vectorstore = None  # no database available yet
    st.sidebar.warning("Please upload your PDF documents above to get started.")
#
#
# After Running the 3rd block of code, Sidebar now shows:
# ----------> "Your Documents" heading
# ----------> Upload button for PDF files
# ----------> "Please upload your PDF documents above to get started" warning
# 
# Two things still showing from before that we noted:
# ----------> "System Status" with "Database loaded: 4050 chunks"  → developer info, clean up later
# ----------> "Deploy" button top right                            → Streamlit's own button, ignore for now
#
# The upload button is working. The warning message is correct and it tells users to upload before asking questions.
# Now we will test the uploader by uploading one of your research papers from the pdfs/ folder.
# Just drag and drop a PDF onto the Upload button in the sidebar is also okay.
#
#
# -------------------------BLOCK 4 | CHAT INTERFACE-------------------------
#
#
# The code in this block adds the conversation display and the question input box to the main area of the page.
# Streamlit has a built-in chat interface taht handles messages bubbles, user and assistant icons, and conversation history automatically.
# "st.session_state" stores the conversation history across interactions.
# without "session_state" every time the user types something, Streamlit reruns the whole script and the conversation disappears.
# We just need to initialize it and display existing messages.
#
# Initialize conversation history in session state if it does not exist yet

if "messages" not in st.session_state:
    st.session_state.messages = []     # empty list to store conversation history

# display all previous messages in the conversation

for message in st.session_state.messages:
    with st.chat_message(message["role"]):     # "user" or "assistant" - sets the icon
        st.markdown(message["content"])        # display message text with markdown formatting

# chat input box at the bottom of the page

question = st.chat_input("Ask a question about your documents...")   # returns None if empty
#
#
# After this stage our websites looks like a real product now.
# ----------------------------------------------------------------------------------
# Chat input box at the bottom         |  "Ask a question about your documents..."
# Conversation area in the middle      |  Empty now, will fill with messages
# Sidebar clean                        |  Just upload button and warning
# ----------------------------------------------------------------------------------
# This is exactly what CHATGPT and CLAUDE looks like with input at the bottom, conversation above.
#
#
# -------------------------BLOCK 5 | HANDLE QUESTIONS AND DISPLAYS ANSWERS-------------------------
#
#
# This is the final piece in this class. When a user types a question this block catches it, checks if the documents are uploaded,
# runs the agent, and displays both the question and answer in the chat interface.
# It also saves everything to the conversation history so the chat persists across interactions.
# IN SHORT
# This runs everytime the user types a question and hits enter.
# It calls the agent, gets the answer, and displays it in the chat.
# It also saves the conversation to "session_state" so history persists
#
#
# What needs to happen when a user types a question.
# ---------------> 1. Did the user actually type something?
# ---------------> 2. Did they upload documents?
# ---------------> 3. Show their question on screen
# ---------------> 4. Get the answer somehow
# ---------------> 5. Show the answer on screen
# ---------------> 6. Save the conversation so it persists
#
#
# Step 1: This block of code answers the question: "Did the user type anything?"" If the input box is empty, skip everything below.
if question:  # only runs if user typed something and hit enter
    
# Step 2: This block of code answers the question: "Did the user upload their documents?" If not, show a warning and stop. No documents means no answers.
    # check if documents are uploaded before answering
    if active_vectorstore is None:
        st.warning("Please upload your PDF documents first using the sidebar.")
    
    else:
        # display user question in chat
# Step 3: This block of code answers the question: "Show the user's question on screen in the chat bubble" and save it to conversation history.
        with st.chat_message("user"):
            st.markdown(question)
        
        # save user question to conversation history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # get answer from agent
# Step 4: This block of code answers the question: "Get the answer." This is the same logic from "agent.py". Checking the math first, thean searching the documents, than falling back to the web if documents are not enough.
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):  # loading indicator while thinking
                
                # build conversation history string for agent context
                history = []
                for msg in st.session_state.messages[:-1]:  # all messages except current question
                    history.append(f"{msg['role'].upper()}: {msg['content']}")
                
                # run the agent logic inline
                # check for math first
                math_operators = ["+", "-", "*", "/", "^"]
                has_operator = any(op in question for op in math_operators)
                has_numbers = any(c.isdigit() for c in question)
                is_short = len(question.split()) < 6
                
                if has_operator and has_numbers and is_short:
                    # pure math expression
                    try:
                        result = eval(question)
                        answer = f"Result: {result}"
                        tool_used = "calculator"
                    except:
                        answer = "Could not calculate that expression."
                        tool_used = "calculator"
                
                else:
                    # search documents first
                    doc_results = active_vectorstore.similarity_search(question, k=4)
                    context = "\n\n".join([doc.page_content for doc in doc_results])
                    sources = list(set([doc.metadata.get("source", "unknown") for doc in doc_results]))
                    
                    # ask LLM to answer from documents
                    first_prompt = f"""You are a research assistant. Answer using ONLY the document excerpts below.
If the documents do not contain enough information, respond with exactly: INSUFFICIENT
Do not include bracketed citation numbers like [39] or [11].

Documents:
{context}

Question: {question}

Answer:"""
                    
                    first_response = ollama.chat(
                        model="llama3.2",
                        messages=[{"role": "user", "content": first_prompt}]
                    )
                    first_answer = first_response["message"]["content"].strip()
                    
                    if "INSUFFICIENT" in first_answer:
                        # fall back to web search
                        tool_used = "web search"
                        web_response = tavily_client.search(query=question, max_results=3)
                        web_results = web_response.get("results", [])
                        web_context = "\n\n".join([r["content"] for r in web_results])
                        
                        web_prompt = f"""Answer this question using the web search results below.
Do not include bracketed citation numbers.

Web results:
{web_context}

Question: {question}

Answer:"""
                        web_answer = ollama.chat(
                            model="llama3.2",
                            messages=[{"role": "user", "content": web_prompt}]
                        )
                        answer = web_answer["message"]["content"]
                        sources = [r["url"] for r in web_results]
                    
                    else:
                        tool_used = "documents"
                        answer = first_answer
                
                # display the answer
# Step 5: This block of code answers the question: "Display the answer on screen and show which source was used. Documents, Web, or Calculator."
                st.markdown(answer)

                # log every question and answer to a file for monitoring
                import datetime
                log_entry = f"""[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]
Q: {question}
Tool: {tool_used}
A: {answer}
---
"""
                with open("logs.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(log_entry)  # append to log file, never overwrite
                
                # show which tool was used and sources
                if tool_used == "documents":
                    st.caption(f"Sources: {', '.join(sources)}")
                elif tool_used == "web search":
                    st.caption(f"Web sources: {', '.join(sources[:2])}")
                else:
                    st.caption("Calculated directly")
        
        # save answer to conversation history
# Step 6: This block of code answers the question: "Save the answers to the conversation history so it persists and stays visible when the next question is asked."
        st.session_state.messages.append({"role": "assistant", "content": answer})
#
#
# After this stage, we will get a fully working product.
# ------------------------------------------------------------------------------------------------------
# User uploads a paper       |  Habchi...v_2014.pdf, 389 chunks
# Asks 3 questions           |  All answered
# Conversation persists      |  All messages visible in chat
# Web fallback working       |  The user asked the question not related to documents and the app fell back to WEB to give a grounded answer
# Chat interface             |  User bubbles and assistant bubbles
# Sources shown              |  Web URLs shown at bottom of answer
# ------------------------------------------------------------------------------------------------------
# The third question that I asked in the app, the document didn't have enough context so it fell back to web search and gave me 
# a detailed answer with cited URLs. Exactly the behavior I designed.
#
# At this point, this is no longer a terminal script. This is a REAL WEB APPLICATION.
#
#
# -------------------------DISCLAIMER FOOTER-------------------------
st.sidebar.caption("Research Assistant is AI and can make mistakes. Please double check responses against the original documents.")