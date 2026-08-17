# -------------------------EVALUATION PHASE | Custom RAGAS-style Scoring-------------------------
# We will import everything we need. Much simpler than before. No RAGAS, just our own tools plus the Anthropic library to call Claude directly.
# Tests our RAG system with known questions, sends results to Claude,
# and scores faithfulness and answer relevancy without external RAGAS dependency.
# Target: above 0.7 on faithfulness before moving to Phase 2.

import os                            # access environment variables
import json                          # parse Claude's structured responses
import ollama                        # for embeddings and RAG answers
import anthropic                     # Anthropic API to use Claude as the judge
from dotenv import load_dotenv       # reads .env file to get API key safely
from langchain_chroma import Chroma  # loads our existing ChromaDB database

# -------------------------LOAD API KEY AND SETUP-------------------------
# Now we load the API key, set up the Anthropic client to talk to Claude, and load our existing ChromaDB database. 
# Same embedding class as always. Must match what we used when building the database.

load_dotenv()                               # reads ".env" file and loads ANTHROPIC_API_KEY into environment

# set up Anthropic client to talk to Claude directly
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")  # load key from .env file safely
)

# Custom embedding class. This is identical to "embed_and_store.py" and "ask_question.py"
class OllamaEmbedder:
    def embed_documents(self, texts):       # embed a list of texts
        return [ollama.embeddings(model="nomic-embed-text", prompt=t)["embedding"] for t in texts]

    def embed_query(self, text):            # embed a single query
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

embeddings = OllamaEmbedder()               # create instance of our custom embedder

# load existing ChromaDB database
vectorstore = Chroma(
    persist_directory="research_db",        # folder where our vectors are stored
    embedding_function=embeddings           # must match model used when database was created
)

print("Database loaded.")
print(f"Total chunks available: {vectorstore._collection.count()}")

# -------------------------RAG FUNCTION-------------------------
# Next, we will write two functions. The first runs a question through our RAG system and returns the answer and retrieved chunks.
# The second asks Claude to judge whether the answer is faithful to the retrieved chunks and whether it is relevant to the question.
# This is the core of our evaluation.


def get_rag_response(question, k=4):                              # runs one question through our RAG pipeline
    
    results = vectorstore.similarity_search(question, k=k)        # find most similar chunks
    context = "\n\n".join([doc.page_content for doc in results])  # join chunks into one block
    
    prompt = f"""You are a scientific research assistant.
Answer the question using ONLY the information provided in the context below.
If the answer is not in the context, say "I could not find this in the provided papers."
Do not include bracketed citation numbers like [39] or [11] in your answer.

Context:
{context}

Question: {question}

Answer:"""
    
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response["message"]["content"]                           # extract answer text
    contexts = [doc.page_content for doc in results]                  # list of retrieved chunks
    sources = list(set([doc.metadata["source"] for doc in results]))  # source paper names
    
    return answer, contexts, sources                                  # return answer, chunks, and paper names

# -------------------------CLAUDE JUDGE FUNCTION-------------------------
def evaluate_with_claude(question, answer, contexts):                 # asks Claude to score one Q&A pair
    
    context_text = "\n\n".join(contexts)                              # join retrieved chunks into one block
    
    # ask Claude to evaluate faithfulness and relevancy
    evaluation_prompt = f"""You are an expert evaluator of AI-generated answers from research papers.

Evaluate the following answer on two criteria and respond with ONLY a JSON object.

Question: {question}

Retrieved context from papers:
{context_text}

Generated answer: {answer}

Evaluate:
1. Faithfulness (0.0 to 1.0): Is the answer based only on the retrieved context? 
   1.0 = completely faithful, every claim is in the context
   0.5 = partially faithful, some claims not in context
   0.0 = not faithful, answer contradicts or ignores context

2. Answer Relevancy (0.0 to 1.0): Does the answer actually address the question?
   1.0 = completely answers the question
   0.5 = partially answers the question
   0.0 = does not answer the question at all

Respond with ONLY this JSON, no other text:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "reasoning": "brief explanation"}}"""

    response = client.messages.create(
        model= "claude-haiku-4-5-20251001",                            # fast and cheap Claude model for evaluation
        max_tokens=500,                                                # short response (200) since we only need a JSON score but later increased (500) to prevent JSON getting cut-off mid response
        messages=[{"role": "user", "content": evaluation_prompt}]
    )
    
  # parse Claude's JSON response
    result_text = response.content[0].text.strip()  # get response text
    
    # sometimes Claude wraps JSON in markdown code blocks — strip those out
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0].strip()
    
    scores = json.loads(result_text)  # convert JSON string to Python dictionary
    
    return scores  # return dictionary with faithfulness and answer_relevancy scores

# -------------------------TEST QUESTIONS-------------------------
# Next, we will write the test questions, run each one through our RAG system, send results to Claude for scoring, 
# calculate the overall average scores, and print a clear pass or fail result.
# Replace these with questions from our human tester with domain knowledge whenever feasible.
# These are placeholders to confirm the evaluation framework works.

test_questions = [
    "What is autophagy and what is its primary function in cells?",
    "What role does BECN1 play in autophagy initiation?",
    "How does the ULK1 complex regulate autophagy?",
    "What is the relationship between autophagy and cancer?",
    "How do cells use autophagy to respond to nutrient starvation?",
    "What is the role of p62 in selective autophagy?",
    "How does mitophagy differ from general autophagy?",
    "What proteins form the autophagy initiation complex?",
    "How does autophagy contribute to cell survival during stress?",
    "What is the relationship between the ubiquitin proteasome system and autophagy?",
]

# -------------------------RUN EVALUATION-------------------------
print(f"\nRunning {len(test_questions)} test questions...")
print("This will take several minutes.\n")

faithfulness_scores = []      # collect faithfulness score for each question
relevancy_scores = []         # collect relevancy score for each question
results_log = []              # collect full results for printing

for i, question in enumerate(test_questions):  # loop through each test question
    print(f"Question {i+1}/{len(test_questions)}: {question[:60]}...")
    
    # get RAG answer and retrieved chunks
    answer, contexts, sources = get_rag_response(question)
    
    # ask Claude to score this answer
    scores = evaluate_with_claude(question, answer, contexts)
    
    faithfulness_scores.append(scores["faithfulness"])    # store faithfulness score
    relevancy_scores.append(scores["answer_relevancy"])   # store relevancy score
    
    results_log.append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "reasoning": scores["reasoning"]
    })
    
    print(f"  Faithfulness: {scores['faithfulness']:.2f} | Relevancy: {scores['answer_relevancy']:.2f}")

# -------------------------PRINT FINAL RESULTS-------------------------
avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)   # calculate average
avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)             # calculate average

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Average Faithfulness:    {avg_faithfulness:.2f}")
print(f"Average Answer Relevancy:{avg_relevancy:.2f}")
print("="*50)

# print detailed results for each question
print("\nDETAILED RESULTS:")
print("-"*50)
for r in results_log:
    print(f"\nQ: {r['question']}")
    print(f"Faithfulness: {r['faithfulness']:.2f} | Relevancy: {r['answer_relevancy']:.2f}")
    print(f"Reasoning: {r['reasoning']}")
    print(f"Sources: {', '.join(r['sources'])}")

# pass or fail decision
print("\n" + "="*50)
if avg_faithfulness >= 0.7:
    print("RESULT: PASS: System is ready for Phase 2")
else:
    print("RESULT: NEEDS TUNING: Faithfulness below 0.7")
    print("Consider: adjusting chunk size, improving system prompt, or increasing k value")
print("="*50)