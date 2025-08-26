#!/usr/bin/env python3
"""
Test script for SQL file processing in multi-format document system
"""

import requests
import json
from pathlib import Path

def test_sql_file_processing():
    """Test SQL file processing via API"""
    
    # API endpoint
    base_url = "http://localhost:8080"
    
    # Test file path
    sql_file_path = "test_files/sample.sql"
    
    print("🧪 Testing SQL File Processing")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Server Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Server not running: {e}")
        return
    
    # Test 2: Check supported formats
    try:
        response = requests.get(f"{base_url}/supported-formats")
        print(f"\n📋 Supported Formats: {response.json()}")
    except Exception as e:
        print(f"❌ Error getting supported formats: {e}")
    
    # Test 3: Test SQL file upload and analysis
    try:
        print(f"\n📁 Testing SQL file: {sql_file_path}")
        
        with open(sql_file_path, 'rb') as f:
            files = {'file': (Path(sql_file_path).name, f, 'application/sql')}
            response = requests.post(f"{base_url}/analyze", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SQL file processed successfully!")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Message: {result.get('message', 'N/A')}")
            
            # Display analysis results
            if 'analysis' in result:
                analysis = result['analysis']
                print(f"\n📊 Analysis Results:")
                print(f"   Author: {analysis.get('Author', 'N/A')}")
                print(f"   Date Created: {analysis.get('DateCreated', 'N/A')}")
                print(f"   Subject: {analysis.get('Subject', 'N/A')}")
                print(f"   Keywords: {analysis.get('Keywords', 'N/A')}")
                print(f"   Summary: {analysis.get('Summary', 'N/A')}")
            
            # Display extracted tables
            if 'tables' in result:
                tables = result['tables']
                print(f"\n📋 Extracted Tables: {len(tables)} found")
                for table_name, table_data in tables.items():
                    print(f"   Table: {table_name}")
                    print(f"   Shape: {table_data.shape if hasattr(table_data, 'shape') else 'N/A'}")
            
            # Display extracted images
            if 'images' in result:
                images = result['images']
                print(f"\n🖼️ Extracted Images: {len(images)} found")
                for source, image_list in images.items():
                    print(f"   Source: {source}")
                    for img in image_list:
                        print(f"     - {img}")
        else:
            print(f"❌ Error processing SQL file: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing SQL file: {e}")
    
    # Test 4: Test text extraction endpoint
    try:
        print(f"\n🔍 Testing text extraction endpoint")
        
        with open(sql_file_path, 'rb') as f:
            files = {'file': (Path(sql_file_path).name, f, 'application/sql')}
            response = requests.post(f"{base_url}/test-extract", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Text extraction successful!")
            print(f"   Extracted text length: {len(result.get('extracted_text', ''))}")
            print(f"   First 200 characters: {result.get('extracted_text', '')[:200]}...")
        else:
            print(f"❌ Error in text extraction: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing text extraction: {e}")

def test_sql_content_analysis():
    """Test direct SQL content analysis"""
    
    print("\n" + "=" * 50)
    print("🔬 Direct SQL Content Analysis")
    print("=" * 50)
    
    # Read SQL file content
    sql_file_path = "test_files/sample.sql"
    
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        print(f"📄 SQL File Content Analysis:")
        print(f"   File size: {len(sql_content)} characters")
        print(f"   Lines: {len(sql_content.splitlines())}")
        
        # Analyze SQL structure
        lines = sql_content.splitlines()
        create_tables = [line for line in lines if 'CREATE TABLE' in line.upper()]
        insert_statements = [line for line in lines if 'INSERT INTO' in line.upper()]
        
        print(f"\n🏗️ SQL Structure:")
        print(f"   CREATE TABLE statements: {len(create_tables)}")
        for table in create_tables:
            print(f"     - {table.strip()}")
        
        print(f"   INSERT statements: {len(insert_statements)}")
        for insert in insert_statements:
            print(f"     - {insert.strip()}")
        
        # Check for table names
        import re
        table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)'
        tables = re.findall(table_pattern, sql_content, re.IGNORECASE)
        print(f"\n📋 Tables found: {tables}")
        
    except Exception as e:
        print(f"❌ Error analyzing SQL content: {e}")

if __name__ == "__main__":
    test_sql_file_processing()
    test_sql_content_analysis()
