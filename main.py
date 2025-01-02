from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uuid
import os
import logging
from typing import Dict, List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import nltk
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CPU-Optimized Document QA System")

# Using lighter, CPU-friendly models
## EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, fast embedding model
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # One of the best performing embedding models
QA_MODEL = "deepset/minilm-uncased-squad2"  # Lightweight QA model

# Initialize components with CPU optimization
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize QA pipeline with CPU settings
qa_pipeline = pipeline('question-answering', 
                      model=QA_MODEL, 
                      device=-1)  # -1 means CPU

# Optimize chunk size for CPU processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Increased chunk size
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", ", ", " ", ""],  # More granular separators
    is_separator_regex=False
)

# In-memory storage
documents: Dict[str, Dict] = {}
vector_stores: Dict[str, FAISS] = {}

class QuestionRequest(BaseModel):
    question: str
    k: int = 3  # Number of relevant chunks to consider

class SearchResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]
    confidence: float
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    # Add space after periods if missing
    text = text.replace(".",". ")
    # Add space after commas if missing
    text = text.replace(",",", ")
    # Remove any non-breaking space characters
    text = text.replace("\xa0", " ")
    return text

def process_document(text: str, asset_id: str) -> FAISS:
    """Process document text into embeddings and store in FAISS."""
    try:
        # Clean the text first
        cleaned_text = clean_text(text)
        
        # Split text into chunks
        chunks = text_splitter.split_text(cleaned_text)
        
        # Clean each chunk
        chunks = [clean_text(chunk) for chunk in chunks]
        
        # Remove empty chunks and very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        # Process chunks in batches
        batch_size = 50
        all_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            all_chunks.extend(batch)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        # Create and store FAISS index
        vector_store = FAISS.from_texts(
            all_chunks,
            embeddings,
            metadatas=[{"chunk_id": i, "asset_id": asset_id} for i in range(len(all_chunks))]
        )
        
        return vector_store
    
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        raise

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file with proper spacing."""
    try:
        from pypdf import PdfReader
        
        text = []
        reader = PdfReader(file_path)
        
        for page in reader.pages:
            # Extract text and clean up spacing
            page_text = page.extract_text()
            # Clean up excessive spaces while preserving legitimate ones
            page_text = ' '.join(page_text.split())
            text.append(page_text)
            
        return "\n\n".join(text)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    
@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document with memory-efficient processing."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        asset_id = str(uuid.uuid4())
        file_path = os.path.join("uploads", f"{asset_id}.pdf")
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)




        # Extract text (using your existing extract_text_from_pdf function)
        text_content = extract_text_from_pdf(file_path)
        
        # Process document with progress logging
        logger.info("Starting document processing...")
        vector_store = process_document(text_content, asset_id)
        vector_stores[asset_id] = vector_store
        
        # Store document info
        documents[asset_id] = {
            "file_path": file_path,
            "original_filename": file.filename
        }

        return {"asset_id": asset_id, "message": "Document processed successfully"}
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/ask/{asset_id}", response_model=SearchResponse)
async def ask_question(asset_id: str, request: QuestionRequest):
    """Enhanced question answering with better summarization."""
    if asset_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        # For summarization requests, increase the number of chunks
        k = request.k
        if "summarize" in request.question.lower() or "summarise" in request.question.lower():
            k = min(k + 5, 8)  # Get more context for summarization
        
        # Retrieve relevant chunks
        vector_store = vector_stores[asset_id]
        relevant_docs = vector_store.similarity_search(
            request.question,
            k=k
        )
        
        # Combine relevant chunks into context
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Adjust prompt based on question type
        if "summarize" in request.question.lower() or "summarise" in request.question.lower():
            question = "Please provide a concise summary of the following text: "
        else:
            question = request.question
            
        # Get answer using QA pipeline
        qa_result = qa_pipeline(
            question=question,
            context=context,
            max_answer_len=150 if "summarize" in request.question.lower() else 50
        )
        
        return SearchResponse(
            answer=qa_result["answer"],
            relevant_chunks=[doc.page_content for doc in relevant_docs],
            confidence=qa_result["score"]
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))