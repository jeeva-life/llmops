"""
Tests for multi-format document processing
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from utils.enhanced_document_processor import (
    EnhancedDocumentProcessor, 
    FastAPIFileAdapter,
    SUPPORTED_EXTENSIONS
)
from src.document_Ingestion.enhanced_data_ingestion import (
    EnhancedDocHandler,
    EnhancedDocumentComparator,
    EnhancedChatIngestor
)


class TestMultiFormatSupport:
    """Test multi-format document processing"""
    
    def test_supported_extensions(self):
        """Test that all required extensions are supported"""
        required_extensions = {".pdf", ".docx", ".txt", ".ppt", ".pptx", ".xlsx", ".csv", ".md", ".sql", ".db"}
        assert required_extensions.issubset(SUPPORTED_EXTENSIONS)
    
    def test_fastapi_file_adapter_validation(self):
        """Test FastAPIFileAdapter validation"""
        # Test supported file
        mock_file = Mock()
        mock_file.filename = "test.pdf"
        mock_file.content_type = "application/pdf"
        
        adapter = FastAPIFileAdapter(mock_file)
        assert adapter.is_supported() is True
        assert adapter.get_extension() == ".pdf"
        
        # Test unsupported file
        mock_file.filename = "test.xyz"
        adapter = FastAPIFileAdapter(mock_file)
        assert adapter.is_supported() is False
    
    def test_enhanced_doc_handler_initialization(self):
        """Test EnhancedDocHandler initialization"""
        handler = EnhancedDocHandler()
        assert handler.session_id is not None
        assert handler.processor is not None
        assert os.path.exists(handler.session_path)
    
    @patch('utils.enhanced_document_processor.EnhancedDocumentProcessor.load_documents')
    def test_enhanced_doc_handler_read_file(self, mock_load_docs):
        """Test reading different file types"""
        handler = EnhancedDocHandler()
        
        # Mock document loading
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_load_docs.return_value = [mock_doc]
        
        # Test PDF reading
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF content")
            pdf_path = f.name
        
        try:
            result = handler.read_file(pdf_path)
            assert "Test content" in result
        finally:
            os.unlink(pdf_path)
    
    def test_enhanced_document_comparator_initialization(self):
        """Test EnhancedDocumentComparator initialization"""
        comparator = EnhancedDocumentComparator()
        assert comparator.session_id is not None
        assert comparator.processor is not None
        assert comparator.session_path.exists()
    
    def test_enhanced_chat_ingestor_initialization(self):
        """Test EnhancedChatIngestor initialization"""
        ingestor = EnhancedChatIngestor()
        assert ingestor.session_id is not None
        assert ingestor.processor is not None
        assert ingestor.faiss_path.exists()
    
    def test_table_extraction(self):
        """Test table extraction from documents"""
        processor = EnhancedDocumentProcessor()
        
        # Create test document with table-like content
        test_content = "Name,Age,City\nJohn,25,NYC\nJane,30,LA"
        mock_doc = Mock()
        mock_doc.page_content = test_content
        mock_doc.metadata = {"source": "test.csv"}
        
        tables = processor.extract_tables_from_documents([mock_doc])
        assert len(tables) > 0
        
        # Check if DataFrame was created
        for table_name, df in tables.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df.columns) > 0
    
    def test_image_extraction(self):
        """Test image extraction from documents"""
        processor = EnhancedDocumentProcessor()
        
        # Create test document with image references
        test_content = "![Image](image.jpg) <img src='photo.png'>"
        mock_doc = Mock()
        mock_doc.page_content = test_content
        mock_doc.metadata = {"source": "test.md"}
        
        images = processor.extract_images_from_documents([mock_doc])
        assert len(images) > 0
        
        # Check if images were found
        for source, image_list in images.items():
            assert len(image_list) > 0
            assert any("image.jpg" in img or "photo.png" in img for img in image_list)
    
    def test_sql_file_processing(self):
        """Test SQL file processing"""
        processor = EnhancedDocumentProcessor()
        
        # Create test SQL file
        sql_content = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        );
        
        INSERT INTO users VALUES (1, 'John', 'john@example.com');
        """
        
        with tempfile.NamedTemporaryFile(suffix=".sql", mode='w', delete=False) as f:
            f.write(sql_content)
            sql_path = Path(f.name)
        
        try:
            docs = processor._process_sql_file(sql_path)
            assert len(docs) > 0
            assert "CREATE TABLE" in docs[0].page_content
            assert "users" in docs[0].metadata.get("tables", [])
        finally:
            os.unlink(sql_path)
    
    def test_sqlite_database_processing(self):
        """Test SQLite database processing"""
        processor = EnhancedDocumentProcessor()
        
        # Create test SQLite database
        import sqlite3
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        try:
            # Create test database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL
                )
            """)
            cursor.execute("INSERT INTO test_table VALUES (1, 'Test', 123.45)")
            conn.commit()
            conn.close()
            
            # Process database
            docs = processor._process_sqlite_database(db_path)
            assert len(docs) > 0
            
            # Check if table information is extracted
            doc_content = docs[0].page_content
            assert "test_table" in doc_content
            assert "id" in doc_content
            assert "name" in doc_content
            
        finally:
            os.unlink(db_path)


class TestAPIEndpoints:
    """Test API endpoints with multi-format support"""
    
    @patch('api.main.EnhancedDocHandler')
    def test_analyze_endpoint_multi_format(self, mock_handler):
        """Test analyze endpoint with different file types"""
        from fastapi import UploadFile
        from api.main import analyze_document
        
        # Mock file
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.xlsx"
        
        # Mock handler
        mock_handler_instance = Mock()
        mock_handler_instance.save_file.return_value = "/tmp/test.xlsx"
        mock_handler_instance.read_file.return_value = "Excel content"
        mock_handler_instance.extract_tables.return_value = {"table1": pd.DataFrame({"col1": [1, 2]})}
        mock_handler_instance.extract_images.return_value = {"test.xlsx": ["image1.jpg"]}
        mock_handler.return_value = mock_handler_instance
        
        # Mock analyzer
        with patch('api.main.DocumentAnalyzer') as mock_analyzer:
            mock_analyzer_instance = Mock()
            mock_analyzer_instance.analyze_document.return_value = {
                "title": "Test Document",
                "summary": "Test summary"
            }
            mock_analyzer.return_value = mock_analyzer_instance
            
            # Test the endpoint
            result = analyze_document(mock_file)
            
            # Verify results
            assert "title" in result.body.decode()
            assert "extracted_tables" in result.body.decode()
            assert "extracted_images" in result.body.decode()
    
    def test_supported_formats_endpoint(self):
        """Test supported formats endpoint"""
        from api.main import get_supported_formats
        
        result = get_supported_formats()
        assert "supported_formats" in result
        assert ".pdf" in result["supported_formats"]
        assert ".xlsx" in result["supported_formats"]
        assert ".csv" in result["supported_formats"]


if __name__ == "__main__":
    pytest.main([__file__])
