#!/usr/bin/env python3
"""
Comprehensive SQLAlchemy Integration Test
Tests both engine-level and ORM-level connectivity
"""

from sqlalchemy import create_engine, text
from config.database import DatabaseConfig
from models.user import user_manager

def test_sqlalchemy_engine():
    """Test SQLAlchemy engine connection"""
    try:
        print("�� Testing SQLAlchemy engine...")
        
        # Create engine
        db_url = DatabaseConfig.get_database_url()
        engine = create_engine(db_url, echo=False)
        
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1 as test"))
            print("✅ Engine connection successful!")
            
        return True
    except Exception as e:
        print(f"❌ Engine connection failed: {e}")
        return False

def test_sqlalchemy_orm():
    """Test SQLAlchemy ORM session"""
    try:
        print("�� Testing SQLAlchemy ORM...")
        
        # Test ORM session
        db = user_manager.SessionLocal()
        try:
            result = db.execute(text("SELECT 1 as test"))
            print("✅ ORM session successful!")
        finally:
            db.close()
            
        return True
    except Exception as e:
        print(f"❌ ORM session failed: {e}")
        return False

def test_table_existence():
    """Test if required tables exist"""
    try:
        print("🔧 Testing table existence...")
        
        engine = create_engine(DatabaseConfig.get_database_url())
        with engine.connect() as connection:
            result = connection.execute(text("SHOW TABLES LIKE 'users'"))
            tables = result.fetchall()
            
            if tables:
                print("✅ Users table exists!")
            else:
                print("📝 Users table will be created when you first run the app")
                
        return True
    except Exception as e:
        print(f"❌ Table check failed: {e}")
        return False

def main():
    print("🚀 SQLAlchemy Integration Test")
    print("=" * 50)
    
    tests = [
        test_sqlalchemy_engine,
        test_sqlalchemy_orm,
        test_table_existence
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    if all(results):
        print("\n🎉 All SQLAlchemy tests passed!")
    else:
        print(f"\n⚠️  {results.count(False)} test(s) failed!")

if __name__ == "__main__":
    main()