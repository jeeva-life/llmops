#!/usr/bin/env python3
# test DataBase Connectivity
"""
Direct MySQL connection test
"""

import pymysql
from config.database import DatabaseConfig

def test_direct_connection():
    """Test direct MySQL connection"""
    try:
        print("🔧 Testing direct MySQL connection...")
        
        # Get connection parameters
        host = DatabaseConfig.MYSQL_HOST
        port = DatabaseConfig.MYSQL_PORT
        user = DatabaseConfig.MYSQL_USER
        password = DatabaseConfig.MYSQL_PASSWORD
        database = DatabaseConfig.MYSQL_DATABASE
        
        print(f"🌐 Connecting to: {host}:{port}")
        print(f"👤 User: {user}")
        print(f"📁 Database: {database}")
        
        # Try direct connection
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4'
        )
        
        print("✅ Direct connection successful!")
        
        # Test a simple query
        cursor = connection.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        print(f"✅ Query test successful: {result}")
        
        # Check if users table exists
        cursor.execute("SHOW TABLES LIKE 'users'")
        tables = cursor.fetchall()
        if tables:
            print("✅ Users table exists!")
        else:
            print("📝 Users table will be created when you first run the app")
        
        cursor.close()
        connection.close()
        
        print("\n🎉 Direct MySQL connection is working!")
        return True
        
    except Exception as e:
        print(f"❌ Direct connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    print("🚀 Direct MySQL Connection Test")
    print("=" * 50)
    
    test_direct_connection()
