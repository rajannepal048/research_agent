# ---------------Step 1: LIBRARY IMPORTS---------------
# Load all PDFs from the pdfs/folder
# split them into chunks and print what those chunk looks like
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------Step 2: VARIABLES SETUP---------------
PDF_FOLDER = "pdfs"      # folders where all pdf files are stored. Change this if you rename the folder
all_docs = []            # empty list that will collect every page from every pdf as we load them one by one

print("Loading PDFs---") # terminal shows that the script has started and is working.
print("-" * 50)          # dashes as a visual divider to make terminal output easier to read

# ---------------Step 3: LOAD ALL PDFs FROM THE pdfs/folder---------------
# Now we will loop through every files in the pdfs folder. For each PDF we build the full path to it, open it,
# and extract all the text from every page. We tag each page with the filename it came from so later we know
# which paper answered the question. Then we add all those pages into our main list and print progress
# so we can see each file loading

for filename in os.listdir(PDF_FOLDER):                  # loops through every file in the pdfs folder one by one
    if filename.endswith(".pdf"):                        # only processes PDF files, ignore anything else in the folder
        filepath = os.path.join(PDF_FOLDER, filename)    # build the complete file path by joining folder and filename
        loader = PyPDFLoader(filepath)                   # create a PDF reader object for this specific file
        docs = loader.load()                             # open the PDF and extract all pages as text documents

        for doc in docs:                                 # loops through each individual page extracted from this PDF
            doc.metadata["source"] = filename            # tag each page with its filename so we know which paper it came from

        all_docs.extend(docs)                            # add all pages from this PDF into the main all_docs list
        print(f"Loaded: {filename} ({len(docs)} pages)") # show which file loaded and how many pages it had

# ---------------Step 4: PRINTING SUMMARY AND CHUNKING---------------
# After loading all the PDFs file we will print a summary total pages as a sanity check. Then we will create the
# splitter tool and use it to cut all those pages into smaller chunks. We print how many chunks were created so we
# can see the pipeline is working correctly

print("-" * 50)                               # visual divider to separate loading output from the summary
print(f"Total pages loaded: {len(all_docs)}") # total pages across all PDFs as a sanity check

print("\nSplitting into chunks...")           # let us know chunking is about to start

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,                         # each chunk will be roughly 500 characters long
    chunk_overlap = 50                        # each chunk shares 50 characters with the next to avoid cutting sentences
)

chunks = splitter.split_documents(all_docs)   # split every page from every PDF into chunks

print(f"Total chunks created: {len(chunks)}") # show how many chunks were created in total
print("-" * 50)                               # visual divider

# ---------------Step 5: SHOW A SAMPLE CHUNK---------------
# We pick one chunk and print it out so we can see with our own eyes that what a pipeline has done to our documents. 
# This is important because we need to see what a chunk actually looks like before trusting the whole system.
# We then print the text content of the chunk and which paper it came from

print("\nSample chunk (chunk number 5):")        # heading so we know what we are looking at
print("-" * 50)                                  # visual divider
print(chunks[4].page_content)                    # print the actual text content of chunk number 5 (indec 4 because Python counts from 0)
print("-" * 50)                                  # visual divider
print(f"Source: {chunks[4].metadata['source']}") # show which PDF this chunk came from
