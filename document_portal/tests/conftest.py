# tests/conftest.py

import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch
from pathlib import Path

# Test fixtures and configuration


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for each test"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_model_loader():
    """Mock ModelLoader for testing"""
    with patch('utils.model_loader.ModelLoader') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock_llm = Mock()
    mock_llm.invoke.return_value = "Mock LLM response"
    return mock_llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing"""
    mock_emb = Mock()
    mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return mock_emb


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test Document) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"


@pytest.fixture
def sample_text_content():
    """Sample text content for testing"""
    return "This is a sample document for testing purposes. It contains multiple sentences to test document processing functionality."


@pytest.fixture
def mock_faiss_index():
    """Mock FAISS index for testing"""
    mock_index = Mock()
    mock_index.similarity_search.return_value = [
        Mock(page_content="Sample document 1", metadata={"source": "doc1.pdf"}),
        Mock(page_content="Sample document 2", metadata={"source": "doc2.pdf"})
    ]
    mock_index.add_documents.return_value = None
    mock_index.save_local.return_value = None
    return mock_index


@pytest.fixture
def mock_chain():
    """Mock LangChain chain for testing"""
    mock_chain = Mock()
    mock_chain.invoke.return_value = {
        "answer": "This is a test answer",
        "sources": ["doc1.pdf", "doc2.pdf"]
    }
    return mock_chain


@pytest.fixture
def test_config():
    """Test configuration dictionary"""
    return {
        "temp_base": "test_data",
        "faiss_base": "test_faiss",
        "use_session_dirs": True,
        "session_id": "test_session_123",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "k": 5
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add unit marker to all tests by default
        if "test_" in item.name:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that might take longer
        if any(keyword in item.name.lower() for keyword in ["llm", "model", "embedding", "faiss"]):
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to API endpoint tests
        if "endpoint" in item.name.lower() or "api" in item.name.lower():
            item.add_marker(pytest.mark.integration)
