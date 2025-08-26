# Sample Document

This is a sample markdown document to test multi-format support.

## Features

- **Document Analysis**: Extract key information from documents
- **Table Processing**: Handle structured data in various formats
- **Image Extraction**: Identify and process images in documents

## Code Example

```python
def analyze_document(file_path):
    """Analyze document and extract key information"""
    processor = EnhancedDocumentProcessor()
    docs = processor.load_documents([file_path])
    return processor.extract_tables_from_documents(docs)
```

## Image Reference

![Sample Image](sample_image.jpg)

## Table Data

| Feature | Status | Priority |
|---------|--------|----------|
| PDF Support | ✅ | High |
| Excel Support | ✅ | High |
| CSV Support | ✅ | Medium |
| Markdown Support | ✅ | Medium |
