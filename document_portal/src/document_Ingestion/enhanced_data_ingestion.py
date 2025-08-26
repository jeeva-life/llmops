"""
Enhanced Data Ingestion Module

Supports multiple file formats including tables and image data processing
"""

from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Union
import fitz  # PyMuPDF
import pandas as pd
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from utils.file_io import generate_session_id, save_uploaded_files
from utils.enhanced_document_processor import (
    EnhancedDocumentProcessor, 
    FastAPIFileAdapter,
    concat_for_analysis, 
    concat_for_comparison
)

# Enhanced supported extensions
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".ppt", ".pptx", 
    ".xlsx", ".csv", ".md", ".sql", ".db"
}

class EnhancedDocHandler:
    """
    Enhanced document handler supporting multiple file formats
    """
    
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        self.processor = EnhancedDocumentProcessor()
        log.info("EnhancedDocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_file(self, uploaded_file) -> str:
        """Save any supported file type"""
        try:
            filename = os.path.basename(uploaded_file.name)
            extension = Path(filename).suffix.lower()
            
            if extension not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {extension}. Supported: {SUPPORTED_EXTENSIONS}")
            
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            
            log.info("File saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
            
        except Exception as e:
            log.error("Failed to save file", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save file: {str(e)}", e) from e

    def read_file(self, file_path: str) -> str:
        """Read any supported file type and return text content"""
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension == ".pdf":
                return self._read_pdf(file_path)
            elif extension in [".docx", ".txt", ".ppt", ".pptx", ".xlsx", ".csv", ".md"]:
                return self._read_structured_file(file_path)
            elif extension in [".sql", ".db"]:
                return self._read_sql_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {extension}")
                
        except Exception as e:
            log.error("Failed to read file", error=str(e), file_path=file_path)
            raise DocumentPortalException(f"Failed to read file: {str(e)}", e) from e

    def _read_pdf(self, pdf_path: Path) -> str:
        """Read PDF file using PyMuPDF"""
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")
            text = "\n".join(text_chunks)
            log.info("PDF read successfully", pdf_path=str(pdf_path), session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read PDF", error=str(e), pdf_path=str(pdf_path), session_id=self.session_id)
            raise

    def _read_structured_file(self, file_path: Path) -> str:
        """Read structured files using enhanced document processor"""
        try:
            docs = self.processor.load_documents([file_path])
            if docs:
                return docs[0].page_content
            else:
                return ""
        except Exception as e:
            log.error("Failed to read structured file", error=str(e), file_path=str(file_path))
            raise

    def _read_sql_file(self, file_path: Path) -> str:
        """Read SQL files and databases"""
        try:
            docs = self.processor.load_documents([file_path])
            if docs:
                return docs[0].page_content
            else:
                return ""
        except Exception as e:
            log.error("Failed to read SQL file", error=str(e), file_path=str(file_path))
            raise

    def extract_tables(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Extract table data from documents"""
        try:
            docs = self.processor.load_documents([Path(file_path)])
            return self.processor.extract_tables_from_documents(docs)
        except Exception as e:
            log.error("Failed to extract tables", error=str(e), file_path=file_path)
            return {}

    def extract_images(self, file_path: str) -> Dict[str, List[str]]:
        """Extract image references from documents"""
        try:
            docs = self.processor.load_documents([Path(file_path)])
            return self.processor.extract_images_from_documents(docs)
        except Exception as e:
            log.error("Failed to extract images", error=str(e), file_path=file_path)
            return {}

class EnhancedDocumentComparator:
    """
    Enhanced document comparator supporting multiple file formats
    """
    
    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.session_id = session_id or generate_session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.processor = EnhancedDocumentProcessor()
        log.info("EnhancedDocumentComparator initialized", session_path=str(self.session_path))

    def save_uploaded_files(self, reference_file, actual_file):
        """Save uploaded files of any supported type"""
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                extension = Path(fobj.name).suffix.lower()
                if extension not in SUPPORTED_EXTENSIONS:
                    raise ValueError(f"Unsupported file type: {extension}. Supported: {SUPPORTED_EXTENSIONS}")
                
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            
            log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
            
        except Exception as e:
            log.error("Error saving files", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error saving files", e) from e

    def read_file(self, file_path: Path) -> str:
        """Read any supported file type"""
        try:
            extension = file_path.suffix.lower()
            
            if extension == ".pdf":
                return self._read_pdf(file_path)
            else:
                docs = self.processor.load_documents([file_path])
                if docs:
                    return docs[0].page_content
                else:
                    return ""
                    
        except Exception as e:
            log.error("Error reading file", file=str(file_path), error=str(e))
            raise DocumentPortalException("Error reading file", e) from e

    def _read_pdf(self, pdf_path: Path) -> str:
        """Read PDF file using PyMuPDF"""
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                parts = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        parts.append(f"\n --- Page {page_num + 1} --- \n{text}")
            log.info("PDF read successfully", file=str(pdf_path), pages=len(parts))
            return "\n".join(parts)
        except Exception as e:
            log.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DocumentPortalException("Error reading PDF", e) from e

    def combine_documents(self) -> str:
        """Combine all documents in session"""
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    content = self.read_file(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
        except Exception as e:
            log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", e) from e

    def extract_tables_from_session(self) -> Dict[str, pd.DataFrame]:
        """Extract tables from all documents in session"""
        tables = {}
        try:
            for file in self.session_path.iterdir():
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    docs = self.processor.load_documents([file])
                    file_tables = self.processor.extract_tables_from_documents(docs)
                    tables.update(file_tables)
            return tables
        except Exception as e:
            log.error("Error extracting tables from session", error=str(e))
            return tables

    def extract_images_from_session(self) -> Dict[str, List[str]]:
        """Extract images from all documents in session"""
        images = {}
        try:
            for file in self.session_path.iterdir():
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    docs = self.processor.load_documents([file])
                    file_images = self.processor.extract_images_from_documents(docs)
                    images.update(file_images)
            return images
        except Exception as e:
            log.error("Error extracting images from session", error=str(e))
            return images

class EnhancedChatIngestor:
    """
    Enhanced chat ingestor supporting multiple file formats
    """
    
    def __init__(
        self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        self.temp_base = Path(temp_base)
        self.faiss_base = Path(faiss_base)
        self.use_session_dirs = use_session_dirs
        self.session_id = session_id or generate_session_id("chat")
        
        if use_session_dirs:
            self.faiss_path = self.faiss_base / self.session_id
        else:
            self.faiss_path = self.faiss_base
        
        self.faiss_path.mkdir(parents=True, exist_ok=True)
        self.processor = EnhancedDocumentProcessor()
        log.info("EnhancedChatIngestor initialized", session_id=self.session_id, faiss_path=str(self.faiss_path))

    def built_retriver(
        self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
    ):
        """Build retriever from uploaded files of any supported type"""
        try:
            # Save uploaded files
            saved_paths = []
            for file in uploaded_files:
                if not hasattr(file, 'name'):
                    continue
                    
                extension = Path(file.name).suffix.lower()
                if extension not in SUPPORTED_EXTENSIONS:
                    log.warning(f"Unsupported file type skipped: {file.name}")
                    continue
                
                # Save file temporarily
                temp_path = self.temp_base / f"{uuid.uuid4().hex}{extension}"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    if hasattr(file, "read"):
                        f.write(file.read())
                    else:
                        f.write(file.getbuffer())
                
                saved_paths.append(temp_path)
            
            # Load documents
            docs = self.processor.load_documents(saved_paths)
            
            if not docs:
                raise DocumentPortalException("No valid documents found", None)
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            split_docs = splitter.split_documents(docs)
            
            # Create FAISS index
            model_loader = ModelLoader()
            embeddings = model_loader.load_embeddings()
            
            self.vs = FAISS.from_documents(split_docs, embeddings)
            self.vs.save_local(str(self.faiss_path), index_name="index")
            
            # Clean up temporary files
            for path in saved_paths:
                if path.exists():
                    path.unlink()
            
            log.info("Enhanced retriever built successfully", 
                    session_id=self.session_id, 
                    docs_loaded=len(docs), 
                    chunks_created=len(split_docs))
            
        except Exception as e:
            log.error("Failed to build enhanced retriever", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build retriever", e) from e

    def extract_tables_and_images(self, uploaded_files: Iterable) -> Dict[str, Any]:
        """Extract tables and images from uploaded files"""
        tables = {}
        images = {}
        
        try:
            for file in uploaded_files:
                if not hasattr(file, 'name'):
                    continue
                    
                extension = Path(file.name).suffix.lower()
                if extension not in SUPPORTED_EXTENSIONS:
                    continue
                
                # Save file temporarily
                temp_path = self.temp_base / f"{uuid.uuid4().hex}{extension}"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    if hasattr(file, "read"):
                        f.write(file.read())
                    else:
                        f.write(file.getbuffer())
                
                # Extract tables and images
                docs = self.processor.load_documents([temp_path])
                file_tables = self.processor.extract_tables_from_documents(docs)
                file_images = self.processor.extract_images_from_documents(docs)
                
                tables.update(file_tables)
                images.update(file_images)
                
                # Clean up
                if temp_path.exists():
                    temp_path.unlink()
            
            return {
                "tables": tables,
                "images": images
            }
            
        except Exception as e:
            log.error("Failed to extract tables and images", error=str(e))
            return {"tables": {}, "images": {}}
