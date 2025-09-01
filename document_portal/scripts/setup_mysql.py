#!/usr/bin/env python3
"""
MySQL Setup Script for Document Portal
Run this script to set up the MySQL database and user
"""

import pymysql
import os
from config.database import DatabaseConfig

def setup_mysql():
    """Set up MySQL database and user"""
    
    # Connect to MySQL as root (you'll need to provide root password)
    print("🔧 Setting up MySQL for Document Portal...")
    
    try:
        # Connect to MySQL server (without specifying database)
        connection = pymysql.connect(
            host=DatabaseConfig.MYSQL_HOST,
            port=DatabaseConfig.MYSQL_PORT,
            user='root',
            password=input("Enter MySQL root password: "),
            charset='utf8mb4'
        )
        
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        print(f"📁 Creating database: {DatabaseConfig.MYSQL_DATABASE}")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DatabaseConfig.MYSQL_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        
        # Create user if it doesn't exist
        print(f"👤 Creating user: {DatabaseConfig.MYSQL_USER}")
        cursor.execute(f"CREATE USER IF NOT EXISTS '{DatabaseConfig.MYSQL_USER}'@'localhost' IDENTIFIED BY '{DatabaseConfig.MYSQL_PASSWORD}'")
        
        # Grant privileges to user
        print("🔑 Granting privileges...")
        cursor.execute(f"GRANT ALL PRIVILEGES ON `{DatabaseConfig.MYSQL_DATABASE}`.* TO '{DatabaseConfig.MYSQL_USER}'@'localhost'")
        cursor.execute("FLUSH PRIVILEGES")
        
        # Test connection with new user
        print("🧪 Testing connection with new user...")
        test_connection = pymysql.connect(
            host=DatabaseConfig.MYSQL_HOST,
            port=DatabaseConfig.MYSQL_PORT,
            user=DatabaseConfig.MYSQL_USER,
            password=DatabaseConfig.MYSQL_PASSWORD,
            database=DatabaseConfig.MYSQL_DATABASE,
            charset='utf8mb4'
        )
        
        print("✅ MySQL setup completed successfully!")
        print(f"📊 Database: {DatabaseConfig.MYSQL_DATABASE}")
        print(f"👤 User: {DatabaseConfig.MYSQL_USER}")
        print(f"🌐 Host: {DatabaseConfig.MYSQL_HOST}:{DatabaseConfig.MYSQL_PORT}")
        
        test_connection.close()
        
    except pymysql.Error as e:
        print(f"❌ MySQL setup failed: {e}")
        return False
    
    finally:
        if 'connection' in locals():
            connection.close()
    
    return True

def create_env_file():
    """Create .env file with MySQL configuration"""
    
    env_content = f"""# MySQL Database Configuration
MYSQL_HOST={DatabaseConfig.MYSQL_HOST}
MYSQL_PORT={DatabaseConfig.MYSQL_PORT}
MYSQL_USER={DatabaseConfig.MYSQL_USER}
MYSQL_PASSWORD={DatabaseConfig.MYSQL_PASSWORD}
MYSQL_DATABASE={DatabaseConfig.MYSQL_DATABASE}

# JWT Secret Key (change this in production!)
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production

# Connection Pool Settings
MYSQL_POOL_SIZE={DatabaseConfig.MYSQL_POOL_SIZE}
MYSQL_MAX_OVERFLOW={DatabaseConfig.MYSQL_MAX_OVERFLOW}
MYSQL_POOL_TIMEOUT={DatabaseConfig.MYSQL_POOL_TIMEOUT}
MYSQL_POOL_RECYCLE={DatabaseConfig.MYSQL_POOL_RECYCLE}
"""
    
    env_file_path = ".env"
    with open(env_file_path, "w") as f:
        f.write(env_content)
    
    print(f"📝 Created {env_file_path} file with MySQL configuration")

if __name__ == "__main__":
    print("🚀 Document Portal MySQL Setup")
    print("=" * 40)
    
    # Check if running from correct directory
    if not os.path.exists("config/database.py"):
        print("❌ Please run this script from the document_portal directory")
        exit(1)
    
    # Setup MySQL
    if setup_mysql():
        create_env_file()
        print("\n🎉 Setup complete! You can now run your application with MySQL.")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the application: uvicorn api.main:app --reload")
        print("3. Access the portal at: http://localhost:8000")
    else:
        print("\n❌ Setup failed. Please check your MySQL configuration.")
