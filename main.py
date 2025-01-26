from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from starlette.responses import JSONResponse
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
from langchain.schema import Document
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Document QA System",
    docs_url=None,
    openapi_url=None,
    root_path="/proxy/8000"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/openapi.json")
async def get_open_api_endpoint():
    openapi_schema = get_openapi(
        title="Advanced Document QA System",
        version="1.0.0",
        description="API for document processing and QA",
        routes=app.routes,
    )
    openapi_schema["servers"] = [
        {"url": "/proxy/8000"}
    ]
    return JSONResponse(openapi_schema)

@app.get("/docs")
async def get_documentation():
    return get_swagger_ui_html(
        openapi_url="./openapi.json",
        title="API Documentation",
        swagger_favicon_url=""
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)

llm = Ollama(model="mistral")

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


summary_chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=SUMMARY_PROMPT,
    combine_prompt=SUMMARY_PROMPT
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", ", ", " ", ""]
)

documents: Dict[str, Dict] = {}
vector_stores: Dict[str, FAISS] = {}

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
    text = ' '.join(text.split())
    text = text.replace(".",". ")
    text = text.replace(",",", ")
    text = text.replace("\xa0", " ")
    return text

def process_document(text: str, asset_id: str) -> FAISS:
    """Process document text into embeddings and store in FAISS."""
    try:
        cleaned_text = clean_text(text)
        
        chunks = text_splitter.split_text(cleaned_text)
        
        chunks = [clean_text(chunk) for chunk in chunks]
        
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        documents = [
            Document(
                page_content=chunk,
                metadata={"chunk_id": i, "asset_id": asset_id}
            ) for i, chunk in enumerate(chunks)
        ]
        
        vector_store = FAISS.from_documents(
            documents,
            embeddings
        )
        
        return vector_store
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        raise

@app.post("/upload/", include_in_schema=True)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        os.makedirs("uploads", exist_ok=True)
        
        asset_id = str(uuid.uuid4())
        file_path = os.path.join("uploads", f"{asset_id}.pdf")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        text_content = extract_text_from_pdf(file_path)
        vector_store = process_document(text_content, asset_id)
        vector_stores[asset_id] = vector_store
        
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
        
        docs, scores = zip(*search_results)
        context = "\n".join(doc.page_content for doc in docs)
        
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
        os.remove(documents[asset_id]["file_path"])
        
        del documents[asset_id]
        del vector_stores[asset_id]
        
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)