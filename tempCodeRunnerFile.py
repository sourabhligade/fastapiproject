from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uuid
import os
from typing import Dict, Optional
import nltk
import logging
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader, PreProcessor, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try downloading if not already done
nltk.download('punkt')

app = FastAPI(title="Document QA System")

# Initialize Haystack components
document_store = InMemoryDocumentStore(use_bm25=True)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)  # Enable GPU if available
retriever = BM25Retriever(document_store)
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,  # Disable if headers/footers contain useful info
    split_by="word",
    split_length=200,  # Smaller chunks for better context
    split_overlap=50   # Increased overlap for better context
)
qa_pipeline = ExtractiveQAPipeline(retriever, reader)

# In-memory storage for documents
documents: Dict[str, Dict] = {}

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file using pdfplumber."""
    try:
        import pdfplumber
        text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text())
        return "\n\n".join(text)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document and store it with a unique asset ID."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Generate unique asset ID
        asset_id = str(uuid.uuid4())
         
        # Save the file
        file_path = os.path.join(UPLOAD_DIR, f"{asset_id}.pdf")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text from PDF
        text_content = extract_text_from_pdf(file_path)
        
        # Preprocess and index the document
        docs = preprocessor.process([
            {
                "content": text_content,
                "meta": {
                    "asset_id": asset_id,
                    "name": file.filename
                }
            }
        ])
        document_store.write_documents(docs)
        
        # Store document information
        documents[asset_id] = {
            "file_path": file_path,
            "original_filename": file.filename
        }
        
        return {"asset_id": asset_id, "message": "Document uploaded successfully"}
    
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")
    
@app.post("/search/{asset_id}")
async def search_document(asset_id: str, question: QuestionRequest):
    """Search for an answer in a specific document."""
    if asset_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Fetch relevant passages using the retriever
        candidate_docs = retriever.retrieve(
            query=question.question,
            filters={"asset_id": asset_id},
            top_k=5  # Retrieve top 5 passages
        )
        
        # Get answers from the reader
        results = reader.predict(
            query=question.question,
            documents=candidate_docs,
            top_k=1  # Return the best answer
        )
        
        if not results["answers"]:
            return {
                "answer": "No answer found",
                "confidence": 0,
                "question": question.question
            }
        
        # Filter low-confidence answers
        answer = results["answers"][0]
        if answer.score < 0.5:  # Confidence threshold
            return {
                "answer": "No confident answer found",
                "confidence": round(answer.score, 4),
                "question": question.question
            }
        
        return {
            "answer": answer.answer,
            "confidence": round(answer.score, 4),
            "question": question.question,
            "context": answer.context
        }
    
    except Exception as e:
        logger.error(f"Error searching document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{asset_id}")
async def delete_document(asset_id: str):
    """Delete a document by its asset ID."""
    if asset_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove file from filesystem
        os.remove(documents[asset_id]["file_path"])
        # Remove from document store
        document_store.delete_documents(filters={"asset_id": asset_id})
        # Remove from in-memory storage
        del documents[asset_id]
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")