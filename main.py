from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uuid
import os
import logging
from typing import Dict, List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import nltk
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Document QA System")

# Initialize models and components
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize LLM (using Ollama for easier setup)
llm = Ollama(model="mistral")

# Create prompt templates
qa_template = """Answer the following question based on the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question:
{question}

Answer:"""

summary_template = """Provide a comprehensive summary of the following text. Include the main points and key details:

Text:
{text}

Summary:"""

QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])
SUMMARY_PROMPT = PromptTemplate(template=summary_template, input_variables=["text"])

# Initialize summarization chain
summary_chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=SUMMARY_PROMPT,
    combine_prompt=SUMMARY_PROMPT
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", ", ", " ", ""]
)

# Storage
documents: Dict[str, Dict] = {}
vector_stores: Dict[str, FAISS] = {}

# Models
class QuestionRequest(BaseModel):
    question: str
    k: int = 6

class SearchResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]
    confidence: float

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        from pypdf import PdfReader
        
        text = []
        reader = PdfReader(file_path)
        
        for page in reader.pages:
            # Extract and clean text
            page_text = page.extract_text()
            # Clean up spacing
            page_text = ' '.join(page_text.split())
            text.append(page_text)
            
        return "\n\n".join(text)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    # Add space after periods if missing
    text = text.replace(".",". ")
    # Add space after commas if missing
    text = text.replace(",",", ")
    # Remove non-breaking space characters
    text = text.replace("\xa0", " ")
    return text
from langchain.schema import Document

def process_document(text: str, asset_id: str) -> FAISS:
    """Process document text into embeddings and store in FAISS."""
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Split text into chunks
        chunks = text_splitter.split_text(cleaned_text)
        
        # Clean each chunk
        chunks = [clean_text(chunk) for chunk in chunks]
        
        # Remove empty or very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        # Create Document objects
        documents = [
            Document(
                page_content=chunk,
                metadata={"chunk_id": i, "asset_id": asset_id}
            ) for i, chunk in enumerate(chunks)
        ]
        
        # Create vector store
        vector_store = FAISS.from_documents(
            documents,
            embeddings
        )
        
        return vector_store
    
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        raise

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        asset_id = str(uuid.uuid4())
        file_path = os.path.join("uploads", f"{asset_id}.pdf")
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Extract and process text
        text_content = extract_text_from_pdf(file_path)
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
    """Answer questions about the document."""
    if asset_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        vector_store = vector_stores[asset_id]
        search_results = vector_store.similarity_search_with_score(
            request.question,
            k=request.k
        )
        
        # Correctly extract documents and scores
        docs, scores = zip(*search_results)
        context = "\n".join(doc.page_content for doc in docs)
        
        # Handle summary request
        is_summary_request = "summarize" in request.question.lower() or "summarise" in request.question.lower()
        
        if is_summary_request:
            response = summary_chain.run(docs)
        else:
            response = llm(QA_PROMPT.format(
                context=context,
                question=request.question
            ))
        
        return SearchResponse(
            answer=response,
            relevant_chunks=[doc.page_content for doc in docs],
            confidence=1 - min(scores) if scores else 0.95
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/documents/{asset_id}")
async def delete_document(asset_id: str):
    """Delete a document and its associated data."""
    if asset_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        # Delete file
        os.remove(documents[asset_id]["file_path"])
        
        # Delete from storage
        del documents[asset_id]
        del vector_stores[asset_id]
        
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)