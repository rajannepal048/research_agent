# In this script we will take the 1529 chunks that we created in the "load_and_chunk.py" and convert
# each one into a list of numbers that captures it's meaning. Then it stores all the numbers in ChromaDB
# on our machine so that they can be searched instantly later.

# We can understand it like "load_and_chunk.py" reads the PDFs, splits into chunks, and prints the results.
# AND "embed_and_store.py" takes those chunks, converts it into numbers, stores in chromaDB, and saves it
# to disk.
# After this script is run we will have a "research_db/" folder that apprears in our folder projects. That
# folder will be our searchable knowledhe base of all the pdfs paper that is converted into a form the AI
# can search in milliseconds.

# ---------------PHASE 1 | Steps 4 & 5: EMBED CHUNKS AND STORE IN CHROMADB---------------
# Takes the chunks created from our PDFs, converts it into a vector (numbers that capture meaning),
# and stores everything in ChromaDB so that we can search them instantly later.
# Only embed new papers and skips one already processed.

# ---------------IMPORT ALL THE NECESSARY LIBRARY FOR EMBEDDING CHUNKS AND STORING---------------

import os                     # lets Python talk to the file system and check if the folder/files exists
import json                   # lets Python read and write JSON files and is used to track processed papers
from langchain_community.document_loaders import PyPDFLoader        # reads PDF files and extracts text
from langchain_text_splitters import RecursiveCharacterTextSplitter # splits text into chunks
import ollama          # direct ollama library. more reliable than langchain-ollama for embeddings
from langchain_chroma import Chroma  # stores and searches our vectors in ChromaDB
from chromadb import Client          # direct ChromaDB client
import chromadb                      # vector database library

# ---------------SETTINGS---------------
# Now we will define all the settings and the file path the script needs in one place at the top. 
# This is one of the good engineering practice because if we ever need to change a folder name or chunk
# size we can change it here once and not scattered throughout the code

PDF_FOLDER = "pdfs"                      # folder where all the PDF files are stored
DB_FOLDER = "research_db"                # folder where ChromaDB will save the vector database
PROCESSED_FILE = "processed_papers.json" # files that tracks which papers have already been embedded
CHUNK_SIZE = 500                         # each chunk will be roughly 500 characters long
CHUNK_OVERLAP = 50                       # each chunk shares 50 characters with the next to avoid cutting sentences
EMBEDDING_MODEL = "nomic-embed-text"     # dedicated embedding model stored locally via Ollama

# ---------------LOAD PROCESSED PAPERS LIST----------
# In this step we will create a function that loads the list of already processed papers from the JSON file.
# If the file doesn't exist yet, meaning this is the first time we run the script and it returns an empty list.
# This is how the script knows which papers have already been embedded and which ones are new.

def get_processed_papers():
    # check if the tracking file exists yet
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, "r") as f:
            return json.load(f)     # load and return the list of already processed filenames
    return []                       # if file doesn't exist yet return empty list which means nothing is processed yet

# ---------------SAVE PROCESSED PAPERS LIST---------------
# Now, here we will create a function that saves the updated list of processed papers back to the JSON file.
# Every time we finish embedding a new paper, we will call this function to record it. This is how the script
# remembers what it has already done across multiple runs
def save_processed_papers(processed):
    with open(PROCESSED_FILE, "w") as f: # open the tracking file in the write mode and creates it if it doesn't exists
        json.dump(processed, f)          # convert the Python list to JSON and write it to the file

# ----------SET UP EMBEDDING MODEL---------------
# Now we will create the embedding model. This is the tool that creates the text into vectors (list of numbers that capture meaning).
# We are using Ollama to do this locally on our machine, same model we already have downloaded

print("Setting up embedding model...") # lets us know that the script has started

# Custom embedding function that uses ollama directly. Bypasses langchain-ollama port issues
class OllamaEmbedder:
    def embed_documents(self, texts):           # embed a list of texts. Used when storing chunks
        return [ollama.embeddings(model="nomic-embed-text", prompt=t)["embedding"] for t in texts]
    
    def embed_query(self, text):                # embed a single query. Used when searching
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

embeddings = OllamaEmbedder()                   # create instance of our custom embedder

# ---------------FIND NEW PAPERS TO PROCESS---------------
# Here we will load the list of already processed papers, then find which PDFs in the folder are new
# meaning not in that list yet. This is the logic that makes the script smart about only processing new papers.

processed = get_processed_papers() # load the list of already embedded papers from JSON file
all_pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")] # get all the PDF filenames in the folder
new_pdfs = [f for f in all_pdfs if f not in processed]               # find only the ones not yet embedded

print(f"Total papers in folder: {len(all_pdfs)}")                    # show how many PDFs exist in total
print(f"Already processed: {len(processed)}")                        # show how many are already in the database
print(f"New papers to embed: {len(new_pdfs)}")                       # show how many need embedding this run

# ---------------LOAD AND EMBED NEW PAPERS---------------
# Now we will check if there are any new papers to process. If yes we will load them, chunk them, and embed them.
# If there are no new papers we will skip this entirely and just load the existing database.
# This is the core logic that will make the script smart.

new_chunks = [] # empty list to hold chunks from new papers only

if new_pdfs:    # only run this bloack if there are new papers to process
    print("\nLoading and chunking new papers.....")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,                      # use the chunk size we defined in settings
        chunk_overlap = CHUNK_OVERLAP                 # use the overlap we defined the settings
    )

    for filename in new_pdfs:                         # loop through only the new papers
        filepath = os.path.join(PDF_FOLDER, filename) # build full file path
        loader = PyPDFLoader(filepath)                # create PDF reader for this file
        docs = loader.load()                          # extract all pages as text

        for doc in docs:                              # loop through each page
            doc.metadata["source"] = filename         # tag each page with its source file name
        
        chunks = splitter.split_documents(docs)       # split this paper into chunks
        new_chunks.extend(chunks)                     # add chunks to our new chunks list
        processed.append(filename)                    # mark this paper as processed
        print(f"Chunked: {filename} ({len(chunks)} chunks)")     # show progress

    print(f"\nTotal new chunks to embed: {len(new_chunks)}")     # show total new chunks
else:
    print("\nNo new papers found. Loading existing database...") # nothing new to process

# ---------------SAVE TO ChromaDB----------
# This is the final block in this script. 
# If we have new chunks we embed them and either create a new ChromaDB database or add them to the existing one.
# Then we save the updated list of processed papers. If there were no new papers, we just load the existing database.

if new_chunks:                                 # only embed and store if we have new chunks to process
    print("\nEmbedding and storing in ChromaDB...")
    print("This may take a few minutes - embedding 1529 chunks for the first time...")

    if os.path.exists(DB_FOLDER):              # if database already exists add new chunks to it
        vectorstore = Chroma(
            persist_directory = DB_FOLDER,     # point to existing database folder
            embedding_function = embeddings    # use same embedding model as before
        )
        vectorstore.add_documents(new_chunks)  # add only the new chunks to existing database
        print("Added new chunks to existing database.")

    else:                                      # database doesn't exist yet - create it from scratch
        vectorstore = Chroma.from_documents(
            new_chunks,                        # all the chunks to embed and store
            embeddings,                        # the embedding model to use
            persist_directory = DB_FOLDER      # where to save the database on disk
        )
        print("Created new database.")
    
    save_processed_papers(processed)           # save updated list so next run knows what's already done
    print(f"\nDone. Database saved to {DB_FOLDER}/")

else:                                          # no new chunks. Just load the existing database.
    vectorstore = Chroma(
        persist_directory = DB_FOLDER,         # load from existing folder
        embedding_function = embeddings        # must use same model used when database was created
    )        
    print("Existing database loaded successfully.")

print("\nChromaDB is ready to use")            # confirm everything is ready.