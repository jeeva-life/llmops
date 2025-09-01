#!/usr/bin/env python3
"""
Database Health Checker
Comprehensive database connectivity and operation testing for:
- MySQL connection
- SQLAlchemy operations
- User table operations
- Authentication queries
"""

import os
import sys
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseHealthChecker:
    def __init__(self):
        self.mysql_config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'user': os.getenv('MYSQL_USER', 'document_user'),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'database': os.getenv('MYSQL_DATABASE', 'document_portal'),
            'charset': 'utf8mb4',
            'autocommit': True
        }
        
        self.connection = None
        self.engine = None
        self.session = None
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
        
    def print_result(self, success, message):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {message}")
        return success
        
    def test_1_environment_variables(self):
        """Test if all required environment variables are set"""
        self.print_header("Environment Variables Check")
        
        required_vars = [
            'MYSQL_HOST',
            'MYSQL_USER',
            'MYSQL_PASSWORD',
            'MYSQL_DATABASE'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            else:
                # Mask password for security
                display_value = value if var != 'MYSQL_PASSWORD' else '*' * len(value)
                print(f"  {var}: {display_value}")
                
        if not missing_vars:
            return self.print_result(True, "All required environment variables are set")
        else:
            return self.print_result(False, f"Missing variables: {', '.join(missing_vars)}")
            
    def test_2_direct_mysql_connection(self):
        """Test direct MySQL connection using pymysql"""
        self.print_header("Direct MySQL Connection")
        
        try:
            self.connection = pymysql.connect(**self.mysql_config)
            
            if self.connection:
                # Test basic query
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT VERSION()")
                    version = cursor.fetchone()
                    
                return self.print_result(True, f"MySQL connection successful. Version: {version[0]}")
            else:
                return self.print_result(False, "Failed to establish MySQL connection")
                
        except Exception as e:
            return self.print_result(False, f"MySQL connection error: {str(e)}")
            
    def test_3_database_structure(self):
        """Test database structure and tables"""
        self.print_header("Database Structure Check")
        
        if not self.connection:
            return self.print_result(False, "No MySQL connection available")
            
        try:
            with self.connection.cursor() as cursor:
                # Check if database exists
                cursor.execute("SHOW DATABASES LIKE %s", (self.mysql_config['database'],))
                if not cursor.fetchone():
                    return self.print_result(False, f"Database '{self.mysql_config['database']}' does not exist")
                    
                # Check if users table exists
                cursor.execute("SHOW TABLES LIKE 'users'")
                if not cursor.fetchone():
                    return self.print_result(False, "Users table does not exist")
                    
                # Check users table structure
                cursor.execute("DESCRIBE users")
                columns = cursor.fetchall()
                
                required_columns = ['id', 'username', 'email', 'password_hash', 'is_active']
                existing_columns = [col[0] for col in columns]
                
                missing_columns = [col for col in required_columns if col not in existing_columns]
                
                if not missing_columns:
                    return self.print_result(True, f"Users table structure correct. Columns: {', '.join(existing_columns)}")
                else:
                    return self.print_result(False, f"Missing columns: {', '.join(missing_columns)}")
                    
        except Exception as e:
            return self.print_result(False, f"Database structure check error: {str(e)}")
            
    def test_4_sqlalchemy_connection(self):
        """Test SQLAlchemy connection and engine"""
        self.print_header("SQLAlchemy Connection")
        
        try:
            # Build connection string
            password = self.mysql_config['password']
            connection_string = (
                f"mysql+pymysql://{self.mysql_config['user']}:{password}"
                f"@{self.mysql_config['host']}/{self.mysql_config['database']}"
                "?charset=utf8mb4"
            )
            
            self.engine = create_engine(connection_string, echo=False)
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()
                
            if test_value and test_value[0] == 1:
                return self.print_result(True, "SQLAlchemy connection successful")
            else:
                return self.print_result(False, "SQLAlchemy connection test failed")
                
        except Exception as e:
            return self.print_result(False, f"SQLAlchemy connection error: {str(e)}")
            
    def test_5_user_operations(self):
        """Test basic user table operations"""
        self.print_header("User Table Operations")
        
        if not self.connection:
            return self.print_result(False, "No MySQL connection available")
            
        try:
            with self.connection.cursor() as cursor:
                # Test SELECT operation
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                # Test INSERT operation (with cleanup)
                test_username = "test_user_health_check"
                test_email = "test_health@example.com"
                
                # Check if test user exists
                cursor.execute("SELECT id FROM users WHERE username = %s", (test_username,))
                existing_user = cursor.fetchone()
                
                if not existing_user:
                    # Insert test user
                    cursor.execute(
                        "INSERT INTO users (username, email, password_hash, is_active) VALUES (%s, %s, %s, %s)",
                        (test_username, test_email, "test_hash", True)
                    )
                    insert_success = cursor.rowcount > 0
                    
                    # Clean up test user
                    cursor.execute("DELETE FROM users WHERE username = %s", (test_username,))
                    cleanup_success = cursor.rowcount > 0
                    
                    if insert_success and cleanup_success:
                        return self.print_result(True, f"User operations successful. Total users: {user_count}")
                    else:
                        return self.print_result(False, "User operations failed")
                else:
                    return self.print_result(True, f"User table accessible. Total users: {user_count}")
                    
        except Exception as e:
            return self.print_result(False, f"User operations error: {str(e)}")
            
    def test_6_authentication_queries(self):
        """Test authentication-related queries"""
        self.print_header("Authentication Queries")
        
        if not self.connection:
            return self.print_result(False, "No MySQL connection available")
            
        try:
            with self.connection.cursor() as cursor:
                # Test user lookup by username
                cursor.execute("SELECT id, username, email, is_active FROM users LIMIT 1")
                user = cursor.fetchone()
                
                if user:
                    user_id, username, email, is_active = user
                    
                    # Test password verification query
                    cursor.execute(
                        "SELECT password_hash FROM users WHERE username = %s AND is_active = %s",
                        (username, True)
                    )
                    password_result = cursor.fetchone()
                    
                    if password_result:
                        return self.print_result(True, f"Authentication queries successful. Test user: {username}")
                    else:
                        return self.print_result(False, "Password verification query failed")
                else:
                    return self.print_result(False, "No users found for authentication testing")
                    
        except Exception as e:
            return self.print_result(False, f"Authentication queries error: {str(e)}")
            
    def test_7_connection_pooling(self):
        """Test SQLAlchemy connection pooling"""
        self.print_header("Connection Pooling")
        
        if not self.engine:
            return self.print_result(False, "No SQLAlchemy engine available")
            
        try:
            # Test multiple connections
            connections = []
            for i in range(3):
                conn = self.engine.connect()
                connections.append(conn)
                
            # Test queries on different connections
            for i, conn in enumerate(connections):
                result = conn.execute(text(f"SELECT {i+1} as test"))
                test_value = result.fetchone()
                if not test_value or test_value[0] != i+1:
                    return self.print_result(False, f"Connection {i+1} query failed")
                    
            # Close connections
            for conn in connections:
                conn.close()
                
            return self.print_result(True, "Connection pooling working correctly")
            
        except Exception as e:
            return self.print_result(False, f"Connection pooling error: {str(e)}")
            
    def test_8_performance_metrics(self):
        """Test basic performance metrics"""
        self.print_header("Performance Metrics")
        
        if not self.connection:
            return self.print_result(False, "No MySQL connection available")
            
        try:
            with self.connection.cursor() as cursor:
                # Test query execution time
                import time
                start_time = time.time()
                
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                execution_time = time.time() - start_time
                
                if execution_time < 1.0:  # Should complete in under 1 second
                    return self.print_result(True, 
                        f"Query performance acceptable. Users: {user_count}, Time: {execution_time:.3f}s")
                else:
                    return self.print_result(False, 
                        f"Query performance slow. Time: {execution_time:.3f}s")
                        
        except Exception as e:
            return self.print_result(False, f"Performance test error: {str(e)}")
            
    def cleanup(self):
        """Clean up resources"""
        if self.connection:
            self.connection.close()
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
            
    def run_all_tests(self):
        """Run all database health checks"""
        print("🚀 STARTING DATABASE HEALTH CHECK")
        print(f"📅 Check started at: {os.popen('date').read().strip()}")
        print(f"🗄️  Database: {self.mysql_config['database']} on {self.mysql_config['host']}")
        
        tests = [
            self.test_1_environment_variables,
            self.test_2_direct_mysql_connection,
            self.test_3_database_structure,
            self.test_4_sqlalchemy_connection,
            self.test_5_user_operations,
            self.test_6_authentication_queries,
            self.test_7_connection_pooling,
            self.test_8_performance_metrics
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"❌ TEST ERROR: {test.__name__}: {str(e)}")
                results.append(False)
                
        # Summary
        passed = sum(results)
        total = len(results)
        
        print(f"\n{'='*60}")
        print(f"📊 HEALTH CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        print(f"📈 Health Score: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 DATABASE IS HEALTHY! All checks passed.")
        elif passed >= total * 0.8:
            print(f"\n⚠️  DATABASE HAS MINOR ISSUES. {total - passed} check(s) failed.")
        else:
            print(f"\n🚨 DATABASE HAS SERIOUS ISSUES. {total - passed} check(s) failed.")
            
        return passed == total

def main():
    """Main health check execution"""
    checker = DatabaseHealthChecker()
    
    try:
        success = checker.run_all_tests()
        checker.cleanup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Health check interrupted by user")
        checker.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        checker.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
