import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.document_Ingestion.enhanced_data_ingestion import (
    EnhancedDocHandler,
    EnhancedDocumentComparator,
    EnhancedChatIngestor,
)
from src.document_Analyzer.data_analysis import DocumentAnalyzer
from src.document_Compare.document_comparator import DocumentComparatorLLM
from src.document_Chat.retrieval import ConversationalRAG
from utils.enhanced_document_processor import FastAPIFileAdapter
from logger import GLOBAL_LOGGER as log
from auth.routes import router as auth_router
from auth.dependencies import get_current_user, optional_current_user, optional_current_user_no_auth
from models.user import User

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include authentication routes
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request, current_user: Optional[User] = Depends(optional_current_user_no_auth)):
    """Serve the main UI - redirect to login if not authenticated"""
    if not current_user:
        log.info("User not authenticated, redirecting to login")
        return RedirectResponse(url="/login")
    
    log.info(f"Serving UI homepage for user: {current_user.username}")
    resp = templates.TemplateResponse("index.html", {"request": request, "user": current_user})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page"""
    log.info("Serving login page")
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}

@app.get("/supported-formats")
def get_supported_formats() -> Dict[str, Any]:
    """Get list of supported file formats"""
    return {
        "supported_formats": list(FastAPIFileAdapter.SUPPORTED_EXTENSIONS),
        "description": "Multi-format document processing including tables and images"
    }

@app.post("/test-extract")
async def test_extract_text(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Test endpoint to extract text from any supported file"""
    try:
        log.info(f"Testing text extraction for: {file.filename}")
        
        # Validate file type
        adapter = FastAPIFileAdapter(file)
        if not adapter.is_supported():
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported: {', '.join(adapter.SUPPORTED_EXTENSIONS)}"
            )
        
        dh = EnhancedDocHandler()
        saved_path = dh.save_file(adapter)
        text = dh.read_file(saved_path)
        
        return {
            "filename": file.filename,
            "text_length": len(text),
            "extracted_text": text[:1000] + "..." if len(text) > 1000 else text,
            "file_type": adapter.get_extension()
        }
    except Exception as e:
        log.exception("Error during text extraction test")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
) -> Any:
    try:
        log.info(f"Received file for analysis: {file.filename}")
        
        # Validate file type
        adapter = FastAPIFileAdapter(file)
        if not adapter.is_supported():
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported: {', '.join(adapter.SUPPORTED_EXTENSIONS)}"
            )
        
        dh = EnhancedDocHandler()
        saved_path = dh.save_file(adapter)
        text = dh.read_file(saved_path)
        
        # Debug: Log the extracted text
        log.info(f"Extracted text length: {len(text)}")
        log.info(f"First 500 characters: {text[:500]}")
        
        # Extract tables and images
        tables = dh.extract_tables(saved_path)
        images = dh.extract_images(saved_path)
        
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        
        # Add extracted data to result
        result["extracted_tables"] = {name: df.to_dict(orient="records") for name, df in tables.items()}
        result["extracted_images"] = images
        
        log.info("Document analysis complete.")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error during document analysis")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(
    reference: UploadFile = File(...), 
    actual: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
) -> Any:
    try:
        log.info(f"Comparing files: {reference.filename} vs {actual.filename}")
        
        # Validate file types
        ref_adapter = FastAPIFileAdapter(reference)
        act_adapter = FastAPIFileAdapter(actual)
        
        if not ref_adapter.is_supported() or not act_adapter.is_supported():
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported: {', '.join(ref_adapter.SUPPORTED_EXTENSIONS)}"
            )
        
        dc = EnhancedDocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(ref_adapter, act_adapter)
        
        combined_text = dc.combine_documents()
        
        # Extract tables and images from both documents
        tables = dc.extract_tables_from_session()
        images = dc.extract_images_from_session()
        
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        
        result = {
            "rows": df.to_dict(orient="records"), 
            "session_id": dc.session_id,
            "extracted_tables": {name: df.to_dict(orient="records") for name, df in tables.items()},
            "extracted_images": images
        }
        
        log.info("Document comparison completed.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
    current_user: User = Depends(get_current_user)
) -> Any:
    try:
        log.info(f"Indexing chat session. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        
        # Validate file types
        wrapped = []
        for f in files:
            adapter = FastAPIFileAdapter(f)
            if not adapter.is_supported():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {f.filename}. Supported: {', '.join(adapter.SUPPORTED_EXTENSIONS)}"
                )
            wrapped.append(adapter)
        
        # Create enhanced chat ingestor
        ci = EnhancedChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        
        # Build retriever
        ci.built_retriver(
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        
        # Extract tables and images
        extracted_data = ci.extract_tables_and_images(wrapped)
        
        result = {
            "session_id": ci.session_id, 
            "k": k, 
            "use_session_dirs": use_session_dirs,
            "extracted_tables": {name: df.to_dict(orient="records") for name, df in extracted_data["tables"].items()},
            "extracted_images": extracted_data["images"]
        }
        
        log.info(f"Index created successfully for session: {ci.session_id}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
    current_user: User = Depends(get_current_user)
) -> Any:
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])
        log.info("Chat query handled successfully.")

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# command for executing the fast api
# uvicorn api.main:app --port 8080 --reload    
#uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload