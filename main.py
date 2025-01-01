from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uuid
import os
import logging
from typing import Dict
import nltk
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader, PreProcessor, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK downloads
nltk.download("punkt")

app = FastAPI(title="Document QA System")

# Initialize Haystack components
document_store = InMemoryDocumentStore(use_bm25=True)
reader = FARMReader(model_name_or_path="deepset/roberta-large-squad2", use_gpu=True)  # Use a more advanced model
retriever = BM25Retriever(document_store)
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=200,
    split_overlap=50
)
qa_pipeline = ExtractiveQAPipeline(retriever, reader)

# In-memory storage for documents
documents: Dict[str, Dict] = {}

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    asset_id: str
    question: str
    feedback: str

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
        asset_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{asset_id}.pdf")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        text_content = extract_text_from_pdf(file_path)
        word_count = len(nltk.word_tokenize(text_content))  # Calculate word count
        page_count = text_content.count("\f") + 1  # Count form feed characters for pages

        docs = preprocessor.process([
            {
                "content": text_content,
                "meta": {
                    "asset_id": asset_id,
                    "name": file.filename,
                    "word_count": word_count,
                    "page_count": page_count
                }
            }
        ])
        document_store.write_documents(docs)

        documents[asset_id] = {
            "file_path": file_path,
            "original_filename": file.filename,
            "word_count": word_count,
            "page_count": page_count
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

    # Check if the question relates to metadata
    if "length" in question.question.lower():
        doc_info = documents[asset_id]
        return {
            "answer": f"The document has {doc_info['word_count']} words and {doc_info['page_count']} pages.",
            "confidence": 1.0,
            "question": question.question
        }

    try:
        candidate_docs = retriever.retrieve(
            query=question.question,
            filters={"asset_id": asset_id},
            top_k=10  # Retrieve more candidates
        )

        results = reader.predict(
            query=question.question,
            documents=candidate_docs,
            top_k=5  # Return multiple answers
        )

        if not results["answers"]:
            return {
                "answer": "No answer found",
                "confidence": 0,
                "question": question.question
            }

        answer = max(results["answers"], key=lambda x: x.score)
        if answer.score < 0.6:  # Confidence threshold
            return {
                "answer": "No confident answer found",
                "confidence": round(answer.score, 4),
                "question": question.question
            }

        return {
            "answer": answer.answer,
            "confidence": round(answer.score, 4),
            "question": question.question,
            "context": answer.context,
            "document_name": candidate_docs[0].meta.get("name", "Unknown")
        }

    except Exception as e:
        logger.error(f"Error searching document: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again later.")

@app.post("/feedback/")
async def submit_feedback(feedback: FeedbackRequest):
    """Collect user feedback."""
    try:
        # For now, simply log feedback
        logger.info(f"Feedback received: {feedback}")
        return {"message": "Thank you for your feedback!"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error submitting feedback")

@app.delete("/documents/{asset_id}")
async def delete_document(asset_id: str):
    """Delete a document by its asset ID."""
    if asset_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        os.remove(documents[asset_id]["file_path"])
        document_store.delete_documents(filters={"asset_id": asset_id})
        del documents[asset_id]
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
