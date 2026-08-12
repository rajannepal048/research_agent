# This is a final piece of Step 6 of Phase 1 of or RAG project. It connects everything that we've built.
# We type a question and this Script converts our question into a vector using nomic-embed-text (same model we used for chunks).
# After that ChromaDB will search all 1529 vectors and finds the 4 most similar chunks. 
# Those 4 chunks will get sent to llama3.2 along with our question and llama3.2 reads the chunks and writes an answer.
# The answer will appera in our terminal along with which paper it came from.
# After this Script works we will have fully functional RAG system.

# --------------------PHASE 1 | Step 6: ASK A QUESTION------------------
# Loads the existing ChromaDB database, takes a question from the user,
# Finds the most relevant chunks, send them to llama 3.2, and returns a grounded answer with citations showing
# which papers the answwer came from.

import ollama # direct ollama library for both embeddings and LLM answers
from langchain_chroma import Chroma # loads our existing ChromaDB vector database

# --------------------LOAD EXISTING DATABASE--------------------
# Here, we will create the same embedding class from "embed_and_store.py" amd use it to load the existing ChromaDB database.
# We must use the exact same embedding model "nomic-embed-text" that we have used when we stored the chunks.
# If we used a different model compare to "embed_and_store.py", our model will start returning garbage results.
# Thus, we are custom embedding class same as "embed_and_store.py" because it must match exactly.

class OllamaEmbedder:
    def embed_documents(self, texts): # embed a list of texts
        return[ollama.embeddings(model="nomic-embed-text", prompt=t)["embedding"] for t in texts]

    def embed_query(self, text):      # embed a single query and this is what gets called when searching
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

embeddings = OllamaEmbedder()         # creates an instance of our custom embedder

# Load the existing ChromaDB database from disk
vectorstore = Chroma(
    persist_directory="research_db",  # same folder where embed_and store.py saved the database
    embedding_function=embeddings     # must use same embedding model used when database was created
)

print("Database Loaded. Ready to answer questions.")
print(f"Total chunks in database: {vectorstore._collection.count()}") # show how many chunks are stored

# --------------------ASK A QUESTION--------------------
# We will write a function that takes a question, searches ChromaDB for the most relevant chunks,
# builds a promt combining those chunks with the question, sends it to llama3.2, and returns the answer
# with source citations. This is the heart of the RAG pipeline, the moment where retrieval and generation comes together.

def ask(question, k=4):    # k=4 means retrieve the 4 most relevant chunks
    
    # Step 1:  Search ChromaDB for most relevant chunks
    results = vectorstore.similarity_search(question, k=k)  # find k chunks most similar to the question

    # Step 2: Build context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in results]) # join all chunks into one block of text

    # Step 3: Build the Prompt
    prompt = f"""You are a scientific research assistant helping a PhD researcher in biochemistry.
    Answer the question using ONLY the information provided in the context below.
    If the answer is not in the context, say "I could not find this in the provided papers."
    Always be precise and cite specific details from the context.
    IMPORTANT: Never include reference numbers like [39], [11], [40] or any bracketed numbers in your answer. These are internal citation markers that are meaningless to the reader. Describe findings and concepts directly without any bracketed numbers.

    Context from research papers:
    {context}

    Question: {question}

    Answer:"""

    # Step 4: Send to llama3.2 and get answer
    response = ollama.chat(
        model="llama3.2",      # use llama3.2 for generating the answer
        messages=[{"role": "user", "content": prompt}]  # send prompt as a user message
    )

    answer = response["message"]["content"] # extract the answer text from the response
    
    import re
    answer = re.sub(r'\[\d+\]', '', answer) # remove citation numbers like [39], [11], [40]
    answer = re.sub(r'\s+', ' ', answer).strip()   # clean up any extra spaces left behind 

    # Step 5: collect source citations
    sources = list(set([doc.metadata["source"] for doc in results])) # get unique paper filenames

    return answer, sources # return both the answer and which paper it came from.

# --------------------MAIN LOOP--------------------
# This is the loop that actually runs the system. It prints a welcome message, waits for the user to type
# a question, calls the "ask" function we just wrote, prints the answer and which papers it came from, then waits
# for the next question. It keeps running until the user types "quit".

print("\n" + "="*50)
print("Research Paper Q&A System")
print("Ask questions about your uploaded research papers and articles.")
print("Type 'quit' to exit.")
print("="*50 + "\n")

while True:                                     # keep running until user types quit
    question = input("Your question: ").strip()  # get question from user, remove extra spaces

    if question.lower() == "quit":              # check if user wants to exit
        print("Goodbye.")
        break                                   # exit the loop

    if not question:                            # check if user pressed enter without typing anything
        print("Please type a question.")
        continue                                # go back to the top of the loop

    print("\nSearching papers...")

    answer, sources = ask(question)             # call our ask function with the question

    print(f"\nAnswer:\n{answer}")               # print the answer
    print(f"\nSources: {','.join(sources)}")    # print which papers answered the question
    print("\n" + "="*50 + "\n")                 # visual divider before next question

