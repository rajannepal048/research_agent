# -------------------------PHASE 2 | AI AGENT LAYER-------------------------
#
# What this script does:
# This is an upgrade from "ask_question.py". Instead of always searching documents
# and answering, this agent DECIDES what to do based on the question.
#
# The agent has three tools:
# ------> Tool 1 - Document Search: searches our ChromaDB database (same as Phase 1)
# ------> Tool 2 - Web Search: searches the internet via Tavily for current information
# ------> Tool 3 - Calculator: evaluates any math expression
#
# The difference from "ask_question.py":
# ------> "ask_question.py" -> fixed pipeline, always searches documents, always answers
# ------> "agent.py" -> dynamic pipeline, agent reads the question and DECIDES which
#                       tool or combination of tools to use
#
# Why we built without LangChain agent framework:
# LangChain 1.3.14 removed initialize_agent and AgentType entirely.
# Rather than chase their updates we built our own agent loop directly.
# Same result, more stable, and we understand every line of it.
#
# This is what AGENTIC AI means — the LLM decides its own next steps
# rather than following a fixed sequence of instructions.

# -------------------------BLOCK 1 | LIBRARY IMPORTS-------------------------
# We import only what we need. No LangChain agent framework.
# We use ollama directly for the LLM, Tavily for web search,
# and ChromaDB for document search. Clean and stable.

import os                           # access environment variables
import json                         # parse structured responses from the LLM
import ollama                       # direct ollama library — used for both embeddings and LLM
from dotenv import load_dotenv      # reads .env file to get API keys safely
from langchain_chroma import Chroma # loads our existing ChromaDB database
from tavily import TavilyClient     # Tavily web search client

# -------------------------BLOCK 2 | LOAD KEYS AND SETUP-------------------------
# Load API keys from ".env" file, set up Tavily client, load ChromaDB database,
# and set up the Ollama LLM. Everything ready before we build the agent loop.

load_dotenv()  # reads .env file and loads all API keys into environment

# set up Tavily web search client
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")  # load Tavily key from .env file
)

# custom embedding class — identical to all other scripts, must match exactly
# if this model changes here it must change everywhere or search breaks
class OllamaEmbedder:
    def embed_documents(self, texts):  # embed a list of texts
        return [ollama.embeddings(model="nomic-embed-text", prompt=t)["embedding"] for t in texts]

    def embed_query(self, text):  # embed a single query
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

embeddings = OllamaEmbedder()  # create instance of our custom embedder

# load existing ChromaDB database from disk
vectorstore = Chroma(
    persist_directory="research_db",  # folder where our vectors are stored
    embedding_function=embeddings     # must match model used when database was created
)

print("Setup complete. Database and tools ready.")
print(f"Total chunks available: {vectorstore._collection.count()}")

# -------------------------BLOCK 3 | DEFINE THE THREE TOOLS-------------------------
# Three plain Python functions. No LangChain Tool wrapper needed.
# Each function takes a string input and returns a string output.
# The agent will call these functions and read their output to form answers.

# Tool 1: DOCUMENT SEARCH
# searches our ChromaDB database for chunks relevant to the query
def search_documents(query: str) -> str:
    results = vectorstore.similarity_search(query, k=4)  # find 4 most similar chunks
    if not results:  # if nothing found return a clear message
        return "No relevant information found in the research documents."
    output = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")       # get paper filename
        output.append(f"From {source}:\n{doc.page_content}") # format with source
    return "\n\n".join(output)  # join all results into one string

# Tool 2: WEB SEARCH
# searches the internet via Tavily for current information not in our documents
def search_web(query: str) -> str:
    response = tavily_client.search(
        query=query,      # the search query
        max_results=3     # return top 3 results to keep context manageable
    )
    results = response.get("results", [])  # extract results list
    if not results:
        return "No web results found."
    output = []
    for r in results:
        output.append(f"Source: {r['url']}\n{r['content']}")  # format with URL
    return "\n\n".join(output)  # join all results into one string

# Tool 3: CALCULATOR
# evaluates a mathematical expression and returns the result
def calculate(expression: str) -> str:
    try:
        result = eval(expression)   # evaluate the math expression
        return f"Result: {result}"  # return formatted result
    except Exception as e:
        return f"Could not calculate: {str(e)}"  # return error if invalid

# tool registry — a dictionary mapping tool names to their functions and descriptions
# the agent reads the descriptions to decide which tool to use for each question
TOOLS = {
    "search_documents": {
        "function": search_documents,
        "description": "Search the research document collection for scientific information, findings, and paper content. Use for questions about research papers, scientific concepts, and academic topics in the uploaded documents."
    },
    "search_web": {
        "function": search_web,
        "description": "Search the internet for current information, recent developments, and anything not covered in the research documents. Use for current events, recent publications, or general knowledge."
    },
    "calculate": {
        "function": calculate,
        "description": "Evaluate mathematical expressions and perform calculations. Use for any math, statistics, ratios, or numerical computations."
    }
}

# -------------------------BLOCK 4 | THE AGENT BRAIN-------------------------
# This is the agent brain. The part that makes this AGENTIC AI. We send the question to llama3.2 along with descriptions of all three tools 
# and ask it to decide which tool to use. The LLM responds with a structured JSON telling us which tool to call and what query to pass to it.
# We then call that tool and send the result back to the LLM to write the final answer.
#
# This is the ReAct pattern meaning #Reason then Act". The LLM reasons about what to do, then acts by calling a tool, then reads the result and answers.
# This is what makes this agentic AI. Instead of always searching documents,
# the LLM reads the question and decides which tool to use.
#
# The pattern is called ReAct: Reason then Act
# Step 1 — send question + tool descriptions to LLM
# Step 2 — LLM decides which tool to use and returns JSON
# Step 3 — we call that tool with the LLM's chosen query
# Step 4 — we send the tool result back to LLM
# Step 5 — LLM writes the final answer using the tool result

def run_agent(question: str, conversation_history: list) -> str:
    
    # Check for pure math expression FIRST before any search
    # Pure math: has operators and numbers and is short (no research words)
    math_operators = ["+", "-", "*", "/", "^"]
    has_operator = any(op in question for op in math_operators)
    has_numbers = any(c.isdigit() for c in question)
    is_short = len(question.split()) < 6
    
    if has_operator and has_numbers and is_short:
        print("Math expression detected. Running calculator...")
        calc_result = calculate(question)
        print("Agent used: 'calculate'")
        return calc_result
    
    # Step 1 — ALWAYS search documents first
    # Product vision: documents are the primary source, web is fallback only
    print("Searching documents first...")
    doc_result = search_documents(question)
    
    # Step 2 — ask LLM to answer from documents
    # If documents are not enough, LLM responds with INSUFFICIENT_CONTEXT
    # This is cleaner than keyword lists — LLM judges relevance naturally
    first_attempt_prompt = f"""You are a research assistant. Answer the question using ONLY the document excerpts below.
If the documents do not contain enough information to answer the question, respond with exactly this word: INSUFFICIENT_CONTEXT
Do not make up information. Do not use outside knowledge.
Do not include bracketed citation numbers like [39] or [11].

Documents:
{doc_result}

Conversation history:
{conversation_history}

Question: {question}

Answer:"""

    first_response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": first_attempt_prompt}]
    )
    
    first_answer = first_response["message"]["content"].strip()
    
    # Step 3 — check if we need web fallback or calculator
    if "INSUFFICIENT" in first_answer:
        # documents not enough — try web search
        print("Documents insufficient. Searching web as fallback...")
        web_result = search_web(question)
        
        web_prompt = f"""You are a research assistant. Answer the question using the web search results below.
Do not include bracketed citation numbers.

Web results:
{web_result}

Question: {question}

Answer:"""

        web_response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": web_prompt}]
        )
        print("Agent used: 'search_web'")
        return web_response["message"]["content"]
    
    # documents were sufficient — return document answer
    print("Agent used: 'search_documents'")
    return first_answer
    
    # Step 3 — check if this is a pure math question
    # Only trigger calculator if question has BOTH math operators AND numbers
    # and does NOT look like a research question
    math_operators = ["+", "-", "*", "/", "^"]
    has_operator = any(op in question for op in math_operators)
    has_numbers = any(c.isdigit() for c in question)
    research_keywords = ["what", "how", "why", "role", "function", "explain", "describe"]
    is_research = any(kw in question.lower() for kw in research_keywords)
    
    if has_operator and has_numbers and not is_research:
        print("Math detected. Running calculator...")
        tool_name = "calculate"
        tool_result = calculate(question)
    
    # Step 4 — send result to LLM to write final answer
    answer_prompt = f"""You are a research assistant. Using the information below, answer the question clearly and precisely.
Do not include bracketed citation numbers like [39] or [11] in your answer.

Source used: {tool_name}
Information found:
{tool_result}

Conversation history:
{conversation_history}

Question: {question}

Answer:"""

    answer_response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": answer_prompt}]
    )
    
    return answer_response["message"]["content"]

# -------------------------BLOCK 5 | CONVERSATION LOOP-------------------------
# The conversation loop. We keep a conversation history list so the agent remembers previous questions and answers.
# Each new question gets the full history so follow-up questions work naturally. Same structure as "ask_question.py" but now the agent decides
# what to do instead of always searching documents.
# We keep conversation history so follow-up questions work naturally.
# The agent sees previous questions and answers before deciding what tool to use.

conversation_history = []  # stores previous questions and answers for context

print("\n" + "="*50)
print("Research Agent")
print("Ask anything — I will search documents, web, or calculate as needed.")
print("Type 'quit' to exit.")
print("="*50 + "\n")

while True:  # keep running until user types quit
    question = input("Your question: ").strip()  # get question from user

    if question.lower() == "quit":  # check if user wants to exit
        print("Goodbye.")
        break

    if not question:  # check if user pressed enter without typing
        print("Please type a question.")
        continue

    print("\nThinking...\n")

    try:
        answer = run_agent(question, conversation_history)  # run the agent

        # add this question and answer to conversation history
        conversation_history.append(f"Q: {question}")
        conversation_history.append(f"A: {answer}")

        # keep history to last 6 exchanges so context does not grow too large
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]

        print(f"\nAnswer: {answer}")  # print the final answer

    except Exception as e:
        print(f"Error: {str(e)}")  # print error if something goes wrong

    print("\n" + "-"*50 + "\n")  # visual divider before next question