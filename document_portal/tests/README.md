# Document Portal Test Suite

This directory contains comprehensive unit tests for the Document Portal project.

## Test Structure

The test suite is organized into several test classes:

### 1. TestFastAPIEndpoints
Tests for all FastAPI endpoints:
- `test_home_endpoint()` - Tests the home page
- `test_health_endpoint()` - Tests the health check endpoint
- `test_analyze_document_success()` - Tests document analysis functionality
- `test_analyze_document_no_file()` - Tests error handling for missing files
- `test_compare_documents_success()` - Tests document comparison
- `test_chat_build_index_success()` - Tests chat index building
- `test_chat_query_success()` - Tests chat query functionality
- `test_chat_query_missing_session_id()` - Tests validation errors

### 2. TestDocumentIngestion
Tests for document ingestion components:
- `test_doc_handler_initialization()` - Tests DocHandler setup
- `test_document_comparator_initialization()` - Tests DocumentComparator setup
- `test_chat_ingestor_initialization()` - Tests ChatIngestor setup
- `test_faiss_manager_initialization()` - Tests FaissManager setup
- `test_faiss_manager_fingerprint_generation()` - Tests fingerprint generation

### 3. TestDocumentAnalysis
Tests for document analysis components:
- `test_document_analyzer_initialization()` - Tests DocumentAnalyzer setup
- `test_document_analyzer_analyze_document()` - Tests document analysis functionality

### 4. TestDocumentComparison
Tests for document comparison components:
- `test_document_comparator_llm_initialization()` - Tests DocumentComparatorLLM setup
- `test_document_comparator_compare_documents()` - Tests comparison functionality

### 5. TestDocumentChat
Tests for document chat components:
- `test_conversational_rag_initialization()` - Tests ConversationalRAG setup
- `test_conversational_rag_load_retriever()` - Tests retriever loading
- `test_conversational_rag_invoke()` - Tests RAG query functionality

### 6. TestUtils
Tests for utility functions:
- `test_fastapi_file_adapter()` - Tests FastAPIFileAdapter
- `test_fastapi_file_adapter_read()` - Tests file reading functionality

### 7. TestErrorHandling
Tests for error handling:
- `test_document_portal_exception()` - Tests custom exception creation
- `test_analyze_document_exception_handling()` - Tests exception handling in analyze endpoint
- `test_compare_documents_exception_handling()` - Tests exception handling in compare endpoint

## Running Tests

### Prerequisites
Install test dependencies:
```bash
pip install pytest pytest-cov pytest-mock
```

### Basic Test Execution
```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_unit_cases.py

# Run specific test class
python -m pytest tests/test_unit_cases.py::TestFastAPIEndpoints

# Run specific test method
python -m pytest tests/test_unit_cases.py::TestFastAPIEndpoints::test_home_endpoint
```

### Using the Test Runner Script
```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --type unit

# Run only integration tests
python run_tests.py --type integration

# Run fast tests (exclude slow ones)
python run_tests.py --type fast

# Run with coverage
python run_tests.py --coverage

# Run with verbose output
python run_tests.py --verbose
```

### Test Markers
Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests (default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.llm` - LLM related tests
- `@pytest.mark.faiss` - FAISS related tests

Run tests by marker:
```bash
# Run only unit tests
python -m pytest -m unit

# Run only integration tests
python -m pytest -m integration

# Run all tests except slow ones
python -m pytest -m "not slow"

# Run API tests only
python -m pytest -m api
```

## Test Fixtures

The `conftest.py` file provides several useful fixtures:

- `test_data_dir` - Temporary directory for test data
- `temp_dir` - Temporary directory for each test
- `mock_model_loader` - Mock ModelLoader
- `mock_llm` - Mock LLM
- `mock_embeddings` - Mock embeddings
- `sample_pdf_content` - Sample PDF content
- `sample_text_content` - Sample text content
- `mock_faiss_index` - Mock FAISS index
- `mock_chain` - Mock LangChain chain
- `test_config` - Test configuration dictionary

## Coverage Reporting

Generate coverage reports:
```bash
# Generate HTML coverage report
python -m pytest --cov=src --cov=api --cov-report=html

# Generate terminal coverage report
python -m pytest --cov=src --cov=api --cov-report=term

# Generate both HTML and terminal reports
python -m pytest --cov=src --cov=api --cov-report=html --cov-report=term
```

## Test Configuration

The `pytest.ini` file configures:
- Test discovery patterns
- Default options
- Markers
- Warning filters

## Best Practices

1. **Mock External Dependencies**: All tests use mocks for external services (LLMs, embeddings, etc.)
2. **Isolated Tests**: Each test is independent and doesn't rely on other tests
3. **Proper Cleanup**: Tests clean up after themselves using fixtures
4. **Descriptive Names**: Test names clearly describe what they're testing
5. **Error Testing**: Tests include both success and error scenarios
6. **Documentation**: Each test has a docstring explaining its purpose

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention: `test_<functionality>_<scenario>()`
2. Add appropriate docstrings
3. Use the provided fixtures when possible
4. Mock external dependencies
5. Test both success and error cases
6. Add appropriate markers for categorization

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running tests from the project root
2. **Mock Issues**: Check that mocks are properly configured for your test
3. **Path Issues**: Use the provided fixtures for temporary directories
4. **Dependency Issues**: Ensure all test dependencies are installed

### Debug Mode
Run tests in debug mode to see more information:
```bash
python -m pytest tests/ -v -s --tb=long
```
