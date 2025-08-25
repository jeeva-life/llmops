# tests/test_unit_cases.py

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
from io import BytesIO
import json
from pathlib import Path

from api.main import app
from src.document_Ingestion.data_ingestion import DocHandler, DocumentComparator, ChatIngestor, FaissManager
from src.document_Analyzer.data_analysis import DocumentAnalyzer
from src.document_Compare.document_comparator import DocumentComparatorLLM
from src.document_Chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter
from exception.custom_exception import DocumentPortalException

client = TestClient(app)


class TestFastAPIEndpoints:
    """Test cases for FastAPI endpoints"""
    
    def test_home_endpoint(self):
        """Test the home page endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Document Portal" in response.text
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "document-portal"
    
    @patch('api.main.DocHandler')
    @patch('api.main.read_pdf_via_handler')
    @patch('api.main.DocumentAnalyzer')
    def test_analyze_document_success(self, mock_analyzer, mock_read_pdf, mock_doc_handler):
        """Test successful document analysis"""
        # Mock setup
        mock_handler = Mock()
        mock_handler.save_pdf.return_value = "/tmp/test.pdf"
        mock_doc_handler.return_value = mock_handler
        
        mock_read_pdf.return_value = "Sample document text"
        
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_document.return_value = {
            "title": "Test Document",
            "summary": "Test summary",
            "key_points": ["point1", "point2"]
        }
        mock_analyzer.return_value = mock_analyzer_instance
        
        # Create test file
        test_content = b"Test PDF content"
        test_file = UploadFile(
            filename="test.pdf",
            file=BytesIO(test_content),
            content_type="application/pdf"
        )
        
        response = client.post("/analyze", files={"file": ("test.pdf", test_content, "application/pdf")})
        
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "summary" in data
    
    def test_analyze_document_no_file(self):
        """Test document analysis with no file"""
        response = client.post("/analyze")
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.DocumentComparator')
    @patch('api.main.DocumentComparatorLLM')
    def test_compare_documents_success(self, mock_comp_llm, mock_doc_comp):
        """Test successful document comparison"""
        # Mock setup
        mock_comp = Mock()
        mock_comp.save_uploaded_files.return_value = ("/tmp/ref.pdf", "/tmp/act.pdf")
        mock_comp.combine_documents.return_value = "Combined document text"
        mock_comp.session_id = "test_session_123"
        mock_doc_comp.return_value = mock_comp
        
        mock_llm = Mock()
        mock_df = Mock()
        mock_df.to_dict.return_value = [{"similarity": 0.8, "section": "intro"}]
        mock_llm.compare_documents.return_value = mock_df
        mock_comp_llm.return_value = mock_llm
        
        # Create test files
        ref_content = b"Reference document content"
        act_content = b"Actual document content"
        
        response = client.post(
            "/compare",
            files={
                "reference": ("ref.pdf", ref_content, "application/pdf"),
                "actual": ("act.pdf", act_content, "application/pdf")
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "rows" in data
        assert "session_id" in data
    
    @patch('api.main.ChatIngestor')
    def test_chat_build_index_success(self, mock_chat_ingestor):
        """Test successful chat index building"""
        # Mock setup
        mock_ingestor = Mock()
        mock_ingestor.session_id = "test_session_456"
        mock_ingestor.built_retriver.return_value = None
        mock_chat_ingestor.return_value = mock_ingestor
        
        # Create test file
        test_content = b"Test document for indexing"
        
        response = client.post(
            "/chat/index",
            files={"files": ("test.pdf", test_content, "application/pdf")},
            data={
                "session_id": "test_session_456",
                "use_session_dirs": "true",
                "chunk_size": "1000",
                "chunk_overlap": "200",
                "k": "5"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["k"] == 5
    
    @patch('api.main.ConversationalRAG')
    @patch('os.path.isdir')
    def test_chat_query_success(self, mock_isdir, mock_rag_class):
        """Test successful chat query"""
        # Mock setup
        mock_isdir.return_value = True
        
        mock_rag = Mock()
        mock_rag.invoke.return_value = "This is the answer to your question"
        mock_rag_class.return_value = mock_rag
        
        response = client.post(
            "/chat/query",
            data={
                "question": "What is this document about?",
                "session_id": "test_session_789",
                "use_session_dirs": "true",
                "k": "5"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "session_id" in data
        assert data["engine"] == "LCEL-RAG"
    
    def test_chat_query_missing_session_id(self):
        """Test chat query with missing session_id when required"""
        response = client.post(
            "/chat/query",
            data={
                "question": "What is this document about?",
                "use_session_dirs": "true",
                "k": "5"
            }
        )
        
        assert response.status_code == 400
        assert "session_id is required" in response.json()["detail"]


class TestDocumentIngestion:
    """Test cases for document ingestion components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.document_Ingestion.data_ingestion.ModelLoader')
    def test_doc_handler_initialization(self, mock_model_loader):
        """Test DocHandler initialization"""
        mock_loader = Mock()
        mock_model_loader.return_value = mock_loader
        
        handler = DocHandler()
        assert handler is not None
    
    @patch('src.document_Ingestion.data_ingestion.ModelLoader')
    def test_document_comparator_initialization(self, mock_model_loader):
        """Test DocumentComparator initialization"""
        mock_loader = Mock()
        mock_model_loader.return_value = mock_loader
        
        comparator = DocumentComparator()
        assert comparator is not None
        assert hasattr(comparator, 'session_id')
    
    @patch('src.document_Ingestion.data_ingestion.ModelLoader')
    def test_chat_ingestor_initialization(self, mock_model_loader):
        """Test ChatIngestor initialization"""
        mock_loader = Mock()
        mock_model_loader.return_value = mock_loader
        
        ingestor = ChatIngestor(
            temp_base=self.temp_dir,
            faiss_base=os.path.join(self.temp_dir, "faiss"),
            use_session_dirs=True,
            session_id="test_session"
        )
        assert ingestor is not None
        assert ingestor.session_id == "test_session"
    
    def test_faiss_manager_initialization(self):
        """Test FaissManager initialization"""
        faiss_dir = os.path.join(self.temp_dir, "faiss")
        manager = FaissManager(faiss_dir)
        assert manager is not None
        assert manager.index_dir == Path(faiss_dir)
    
    def test_faiss_manager_fingerprint_generation(self):
        """Test FaissManager fingerprint generation"""
        faiss_dir = os.path.join(self.temp_dir, "faiss")
        manager = FaissManager(faiss_dir)
        
        text = "Sample text"
        metadata = {"source": "test.pdf", "row_id": "123"}
        fingerprint = manager._fingerprint(text, metadata)
        
        assert fingerprint == "test.pdf::123"
    
    def test_faiss_manager_fingerprint_without_metadata(self):
        """Test FaissManager fingerprint generation without metadata"""
        faiss_dir = os.path.join(self.temp_dir, "faiss")
        manager = FaissManager(faiss_dir)
        
        text = "Sample text"
        metadata = {}
        fingerprint = manager._fingerprint(text, metadata)
        
        # Should generate hash-based fingerprint
        assert len(fingerprint) == 64  # SHA256 hex length


class TestDocumentAnalysis:
    """Test cases for document analysis components"""
    
    @patch('src.document_Analyzer.data_analysis.ModelLoader')
    @patch('src.document_Analyzer.data_analysis.PROMPT_REGISTRY')
    def test_document_analyzer_initialization(self, mock_prompt_registry, mock_model_loader):
        """Test DocumentAnalyzer initialization"""
        mock_loader = Mock()
        mock_llm = Mock()
        mock_loader.load_llm.return_value = mock_llm
        mock_model_loader.return_value = mock_loader
        
        mock_prompt = Mock()
        mock_prompt_registry.__getitem__.return_value = mock_prompt
        
        analyzer = DocumentAnalyzer()
        assert analyzer is not None
        assert analyzer.llm == mock_llm
    
    @patch('src.document_Analyzer.data_analysis.ModelLoader')
    @patch('src.document_Analyzer.data_analysis.PROMPT_REGISTRY')
    def test_document_analyzer_analyze_document(self, mock_prompt_registry, mock_model_loader):
        """Test document analysis functionality"""
        # Mock setup
        mock_loader = Mock()
        mock_llm = Mock()
        mock_loader.load_llm.return_value = mock_llm
        mock_model_loader.return_value = mock_loader
        
        mock_prompt = Mock()
        mock_prompt_registry.__getitem__.return_value = mock_prompt
        
        mock_parser = Mock()
        mock_parser.get_format_instructions.return_value = "Format instructions"
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = {
            "title": "Test Document",
            "summary": "Test summary",
            "key_points": ["point1", "point2"]
        }
        
        with patch('src.document_Analyzer.data_analysis.JsonOutputParser') as mock_json_parser, \
             patch('src.document_Analyzer.data_analysis.OutputFixingParser') as mock_fixing_parser:
            
            mock_json_parser.return_value = mock_parser
            mock_fixing_parser.from_llm.return_value = mock_parser
            
            # Mock the chain creation
            with patch.object(DocumentAnalyzer, '_create_chain', return_value=mock_chain):
                analyzer = DocumentAnalyzer()
                result = analyzer.analyze_document("Sample document text")
                
                assert result["title"] == "Test Document"
                assert result["summary"] == "Test summary"
                assert "key_points" in result


class TestDocumentComparison:
    """Test cases for document comparison components"""
    
    @patch('src.document_Compare.document_comparator.ModelLoader')
    def test_document_comparator_llm_initialization(self, mock_model_loader):
        """Test DocumentComparatorLLM initialization"""
        mock_loader = Mock()
        mock_model_loader.return_value = mock_loader
        
        comparator = DocumentComparatorLLM()
        assert comparator is not None
    
    @patch('src.document_Compare.document_comparator.ModelLoader')
    def test_document_comparator_compare_documents(self, mock_model_loader):
        """Test document comparison functionality"""
        # Mock setup
        mock_loader = Mock()
        mock_model_loader.return_value = mock_loader
        
        comparator = DocumentComparatorLLM()
        
        # Mock the comparison logic
        with patch.object(comparator, '_perform_comparison') as mock_compare:
            mock_df = Mock()
            mock_compare.return_value = mock_df
            
            result = comparator.compare_documents("Combined document text")
            assert result == mock_df


class TestDocumentChat:
    """Test cases for document chat components"""
    
    def test_conversational_rag_initialization(self):
        """Test ConversationalRAG initialization"""
        rag = ConversationalRAG(session_id="test_session")
        assert rag is not None
        assert rag.session_id == "test_session"
    
    @patch('src.document_Chat.retrieval.FAISS')
    def test_conversational_rag_load_retriever(self, mock_faiss):
        """Test ConversationalRAG retriever loading"""
        rag = ConversationalRAG(session_id="test_session")
        
        mock_index = Mock()
        mock_faiss.load_local.return_value = mock_index
        
        with patch('os.path.exists', return_value=True):
            rag.load_retriever_from_faiss("/tmp/test_index", k=5)
            assert rag.retriever is not None
    
    def test_conversational_rag_invoke(self):
        """Test ConversationalRAG invoke method"""
        rag = ConversationalRAG(session_id="test_session")
        
        # Mock the chain
        mock_chain = Mock()
        mock_chain.invoke.return_value = "This is the answer"
        rag.chain = mock_chain
        
        result = rag.invoke("What is this about?", chat_history=[])
        assert result == "This is the answer"


class TestUtils:
    """Test cases for utility functions"""
    
    def test_fastapi_file_adapter(self):
        """Test FastAPIFileAdapter"""
        # Create a mock UploadFile
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.file = BytesIO(b"test content")
        
        adapter = FastAPIFileAdapter(mock_file)
        assert adapter.filename == "test.pdf"
        assert adapter.file == mock_file.file
    
    def test_fastapi_file_adapter_read(self):
        """Test FastAPIFileAdapter read method"""
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.file = BytesIO(b"test content")
        
        adapter = FastAPIFileAdapter(mock_file)
        content = adapter.read()
        assert content == b"test content"


class TestErrorHandling:
    """Test cases for error handling"""
    
    def test_document_portal_exception(self):
        """Test custom exception creation"""
        exception = DocumentPortalException("Test error message", None)
        assert str(exception) == "Test error message"
    
    @patch('api.main.DocHandler')
    def test_analyze_document_exception_handling(self, mock_doc_handler):
        """Test exception handling in analyze endpoint"""
        mock_handler = Mock()
        mock_handler.save_pdf.side_effect = Exception("Test error")
        mock_doc_handler.return_value = mock_handler
        
        test_content = b"Test content"
        response = client.post("/analyze", files={"file": ("test.pdf", test_content, "application/pdf")})
        
        assert response.status_code == 500
        assert "Analysis failed" in response.json()["detail"]
    
    @patch('api.main.DocumentComparator')
    def test_compare_documents_exception_handling(self, mock_doc_comp):
        """Test exception handling in compare endpoint"""
        mock_comp = Mock()
        mock_comp.save_uploaded_files.side_effect = Exception("Test error")
        mock_doc_comp.return_value = mock_comp
        
        ref_content = b"Reference content"
        act_content = b"Actual content"
        
        response = client.post(
            "/compare",
            files={
                "reference": ("ref.pdf", ref_content, "application/pdf"),
                "actual": ("act.pdf", act_content, "application/pdf")
            }
        )
        
        assert response.status_code == 500
        assert "Comparison failed" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])