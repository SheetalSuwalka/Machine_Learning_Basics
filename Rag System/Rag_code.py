"""
End-to-End RAG System
Supports: PDF, PNG (OCR), DOCX
Embedding Model: text-embedding-3-large (OpenAI)
LLM: GPT-4o (OpenAI)
Vector Store: ChromaDB
"""

# ─────────────────────────────────────────────
# Install dependencies (run once):
# pip install openai chromadb pymupdf python-docx pytesseract pillow tiktoken
# Also install Tesseract OCR engine on your system:
#   Ubuntu: sudo apt install tesseract-ocr
#   Mac:    brew install tesseract
#   Windows: https://github.com/UB-Mannheim/tesseract/wiki
# ─────────────────────────────────────────────

import os
import uuid
import fitz                         # PyMuPDF — for PDF parsing
import pytesseract                  # OCR — for PNG/image parsing
from PIL import Image               # Pillow — image handling
from docx import Document           # python-docx — for DOCX parsing
import chromadb                     # Vector store
from chromadb.utils import embedding_functions
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

GEMINI_API_KEY      = "your-gemini-api-key"            # 🔑 Replace with your Gemini key
EMBEDDING_MODEL     = "models/text-embedding-004"       # Gemini embedding model
LLM_MODEL           = "gemini-1.5-pro"                  # Gemini LLM model
CHROMA_COLLECTION   = "rag_documents"
CHROMA_DB_PATH      = "./chroma_db"
CHUNK_SIZE          = 500
CHUNK_OVERLAP       = 100
TOP_K_RESULTS       = 5
DOCS_FOLDER         = "./documents"



# ─────────────────────────────────────────────
# INITIALIZE CLIENTS
# ─────────────────────────────────────────────

import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Custom Gemini Embedding Function for ChromaDB
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"   # Use "retrieval_query" at query time
            )
            embeddings.append(result["embedding"])
        return embeddings

# ChromaDB with Gemini embedding function
gemini_ef = GeminiEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=gemini_ef,
    metadata={"hnsw:space": "cosine"}
)


# ─────────────────────────────────────────────
# STEP 1: DOCUMENT PARSERS
# ─────────────────────────────────────────────

def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    print(f"  ✅ Parsed PDF: {file_path} ({len(text)} chars)")
    return text


def parse_png(file_path: str) -> str:
    """Extract text from an image using Tesseract OCR."""
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    print(f"  ✅ Parsed PNG (OCR): {file_path} ({len(text)} chars)")
    return text


def parse_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    print(f"  ✅ Parsed DOCX: {file_path} ({len(text)} chars)")
    return text


def parse_document(file_path: str) -> str:
    """Route to the correct parser based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return parse_png(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    else:
        print(f"  ⚠️  Unsupported file type: {file_path}")
        return ""


# ─────────────────────────────────────────────
# STEP 2: TEXT CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks for better retrieval context.
    Overlap ensures that sentences split across chunk boundaries are still retrievable.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # Move forward with overlap
    return chunks


# ─────────────────────────────────────────────
# STEP 3: LOAD DOCUMENTS & STORE IN CHROMADB
# ─────────────────────────────────────────────

def load_and_index_documents(docs_folder: str):
    """
    Parse all documents in the folder, chunk them,
    and store embeddings in ChromaDB.
    """
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        print(f"📁 Created documents folder: {docs_folder}")
        print("   Please add your PDF, PNG, DOCX files there and re-run.")
        return

    files = [
        f for f in os.listdir(docs_folder)
        if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".docx"))
    ]

    if not files:
        print("⚠️  No supported documents found in the folder.")
        return

    print(f"\n📂 Found {len(files)} document(s). Parsing and indexing...\n")

    for file_name in files:
        file_path = os.path.join(docs_folder, file_name)

        # Parse the document
        raw_text = parse_document(file_path)
        if not raw_text.strip():
            print(f"  ⚠️  No text extracted from {file_name}, skipping.")
            continue

        # Chunk the text
        chunks = chunk_text(raw_text)
        print(f"  📄 {file_name} → {len(chunks)} chunks")

        # Prepare data for ChromaDB
        ids        = [str(uuid.uuid4()) for _ in chunks]
        metadatas  = [{"source": file_name, "chunk_index": i} for i, _ in enumerate(chunks)]

        # Store in ChromaDB (embeddings are auto-generated by openai_ef)
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    print(f"\n✅ Indexing complete! Total chunks in DB: {collection.count()}\n")


# ─────────────────────────────────────────────
# STEP 4: RETRIEVAL
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:

    # Generate query embedding with correct task type
    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"    # 👈 Different task type than indexing
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"]
        })

    return chunks

# ─────────────────────────────────────────────
# STEP 5: GENERATION (RAG)
# ─────────────────────────────────────────────

def ask_question(question: str) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Build a prompt with the context
    3. Send to GPT-4o and return the answer
    """
    print(f"\n🔍 Question: {question}")

    # Retrieve relevant context
    relevant_chunks = retrieve_relevant_chunks(question)

    if not relevant_chunks:
        return "❌ No relevant information found in the documents."

    # Build context string
    context = ""
    for i, chunk in enumerate(relevant_chunks):
        context += f"\n--- Chunk {i+1} (Source: {chunk['source']}) ---\n{chunk['text']}\n"

    # Print sources used
    sources = list(set([c["source"] for c in relevant_chunks]))
    print(f"📎 Sources used: {', '.join(sources)}")

    # Build the prompt
    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided context.
If the answer is not found in the context, say "I don't have enough information to answer this question."
Always be concise, accurate, and cite the source document when possible.

Context from documents:
{context}

Question: {question}

Answer based on the context above:"""

    # Call Gemini LLM
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1000
        )
    )

    return response.text


# ─────────────────────────────────────────────
# STEP 6: INTERACTIVE Q&A LOOP
# ─────────────────────────────────────────────

def interactive_qa():
    """Run an interactive question-answering loop in the terminal."""
    print("\n" + "="*60)
    print("        💬 RAG Q&A System — Ask your documents!")
    print("="*60)
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        answer = ask_question(question)
        print(f"\n🤖 Answer:\n{answer}\n")
        print("-" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # STEP 1: Index documents (only needs to run once)
    # Comment this out after first run if documents haven't changed
    load_and_index_documents(DOCS_FOLDER)

    # STEP 2: Start interactive Q&A
    if collection.count() > 0:
        interactive_qa()
    else:
        print("⚠️  No documents indexed. Please add files to the './documents' folder and re-run.")