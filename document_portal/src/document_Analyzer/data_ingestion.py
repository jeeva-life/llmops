import os
import fitz # wrapper for PyMuPDF
import uuid
from datetime import datetime, timezone
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
# PyPDF loader import for alternative PDF reading
from langchain_community.document_loaders import PyPDFLoader

class DocumentHandler:
    """
    Handles PDF saving and reading operations.
    Automatically logs all actions and supports session-based operations.
    S3(storage class): Hot, Cold, warm storage -- Data Archive -- life-time period
    This is a COLD storage
    This is Data Archival Strategy

    For every execution, we need to create a new session.
    For every session, we need to create a new directory.
    For every directory, we need to create a new file.
    For every file, we need to create a new object.
    For every object, we need to create a new metadata.
    For every metadata, we need to create a new embedding.
    For every embedding, we need to create a new vector.


    """
    def __init__(self, data_dir=None,session_id=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH",
                os.path.join(os.getcwd(), "dummy_data", "document_analysis")
            )
            self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            #create base session directory
            self.session_path = os.path.join(self.data_dir, self.session_id)

            os.makedirs(self.session_path, exist_ok=True)

            self.log.info("PDFHandler Initialized", session_id=self.session_id, session_path=self.session_path)
            
        except Exception as e:
            self.log.error(f"Error initializing PDFHandler: {str(e)}")
            raise DocumentPortalException(f"Error initializing PDFHandler: {str(e)}", e) from e
        


    def save_pdf(self, uploaded_file, metadata=None):
        """
        Save a PDF file to the session directory.
        """
        try:
            filename = os.path.basename(uploaded_file.name)

            if not filename.lower().endswith(".pdf"):
                raise DocumentPortalException("Invalid file type. Only PDF files are allowed.")
            
            save_path = os.path.join(self.session_path, filename)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            self.log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)

            return save_path
        
        except Exception as e:
            self.log.error(f"Error saving PDF: {str(e)}")
            raise

    def read_pdf(self, pdf_path: str) -> str:
        """
        Read a PDF file and return its content.
        Uses PyPDFLoader from LangChain for PDF processing.
        """
        try:
            # PyPDFLoader implementation
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Extract text from all pages
            text_chunks = []
            for i, doc in enumerate(documents, start=1):
                page_text = doc.page_content
                text_chunks.append(f"\n--- Page {i} ---\n{page_text}")
            
            text = "\n".join(text_chunks)
            
            self.log.info("PDF read successfully using PyPDFLoader", 
                         pdf_path=pdf_path,
                         session_id=self.session_id, 
                         pages=len(text_chunks))
            
            return text
            
        except Exception as e:
            self.log.error(f"Error reading PDF with PyPDFLoader: {str(e)}")
            raise DocumentPortalException(f"Error reading PDF with PyPDFLoader", str(e)) from e
    
    def read_pdf_pymupdf(self, pdf_path: str) -> str:
        """
        Original PyMuPDF implementation (commented out from main read_pdf function).
        Read a PDF file and return its content using PyMuPDF (fitz).
        """
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text_chunks.append(f"\n--- Page {page_num} ---\n{page.get_text()}")
            text="\n".join(text_chunks)

            self.log.info("PDF read successfully using PyMuPDF", pdf_path=pdf_path,
                          session_id=self.session_id, pages=len(text_chunks))
            
            return text
        except Exception as e:
            self.log.error(f"Error reading PDF with PyMuPDF: {str(e)}")
            raise DocumentPortalException(f"Error reading PDF with PyMuPDF", str(e)) from e
        

if __name__ == "__main__":
    from pathlib import Path
    from io import BytesIO # process the file in memory for faster processing

    pdf_path=r"D:\Jeevan\llmops\document_portal\dummy_data\document_analysis\2507.23488v1.pdf"
    class DummyFile:
        def __init__(self,file_path):
            self.name = Path(file_path).name
            self._file_path = file_path
        def getbuffer(self):
            return open(self._file_path, "rb").read()
        
    dummy_pdf = DummyFile(pdf_path)
    
    handler = DocumentHandler()
    try:
        saved_path = handler.save_pdf(dummy_pdf)
        print(f"PDF saved to: {saved_path}")

        # Test PyPDFLoader implementation
        print("\n=== Testing PyPDFLoader Implementation ===")
        content = handler.read_pdf(saved_path)
        print(f"PDF content (PyPDFLoader): {content[:500]}...")

        # Test original PyMuPDF implementation
        print("\n=== Testing PyMuPDF Implementation ===")
        content_pymupdf = handler.read_pdf_pymupdf(saved_path)
        print(f"PDF content (PyMuPDF): {content_pymupdf[:500]}...")

    except Exception as e:
        print(f"Error: {str(e)}")