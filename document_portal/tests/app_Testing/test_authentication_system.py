#!/usr/bin/env python3
"""
Comprehensive Authentication System Test Suite
Tests all aspects of the authentication system including:
- User registration
- User login
- JWT token validation
- Cookie authentication
- Session management
- Protected route access
"""

import os
import sys
import requests
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_USER = {
    "username": "test_user_auth",
    "email": "test_auth@example.com",
    "password": "test_password_123",
    "first_name": "Test",
    "last_name": "User"
}

class AuthenticationSystemTester:
    def __init__(self):
        self.session = requests.Session()
        self.access_token = None
        self.user_data = None
        
    def print_test_header(self, test_name):
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {test_name}")
        print(f"{'='*60}")
        
    def print_result(self, success, message):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {message}")
        return success
        
    def test_1_server_connectivity(self):
        """Test if the FastAPI server is running and accessible"""
        self.print_test_header("Server Connectivity")
        
        try:
            response = self.session.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                return self.print_result(True, "Server is running and accessible")
            else:
                return self.print_result(False, f"Server responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            return self.print_result(False, "Cannot connect to server. Is it running?")
            
    def test_2_user_registration(self):
        """Test user registration endpoint"""
        self.print_test_header("User Registration")
        
        try:
            response = self.session.post(
                f"{BASE_URL}/auth/register",
                json=TEST_USER
            )
            
            if response.status_code == 200:
                result = response.json()
                self.user_data = result
                return self.print_result(True, f"User registered successfully: {result['username']}")
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                return self.print_result(False, f"Registration failed: {error_detail}")
                
        except Exception as e:
            return self.print_result(False, f"Registration error: {str(e)}")
            
    def test_3_user_login(self):
        """Test user login endpoint"""
        self.print_test_header("User Login")
        
        try:
            response = self.session.post(
                f"{BASE_URL}/auth/login",
                json={
                    "username": TEST_USER["username"],
                    "password": TEST_USER["password"]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result['access_token']
                
                # Check if cookie was set
                cookies = response.cookies
                has_auth_cookie = 'access_token' in cookies
                
                return self.print_result(True, 
                    f"Login successful. Token: {self.access_token[:20]}... "
                    f"Cookie set: {has_auth_cookie}")
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                return self.print_result(False, f"Login failed: {error_detail}")
                
        except Exception as e:
            return self.print_result(False, f"Login error: {str(e)}")
            
    def test_4_jwt_token_validation(self):
        """Test JWT token validation"""
        self.print_test_header("JWT Token Validation")
        
        if not self.access_token:
            return self.print_result(False, "No access token available")
            
        try:
            # Test with Authorization header
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(f"{BASE_URL}/auth/me", headers=headers)
            
            if response.status_code == 200:
                user_info = response.json()
                return self.print_result(True, 
                    f"Token valid. User: {user_info['username']}")
            else:
                return self.print_result(False, f"Token validation failed: {response.status_code}")
                
        except Exception as e:
            return self.print_result(False, f"Token validation error: {str(e)}")
            
    def test_5_cookie_authentication(self):
        """Test cookie-based authentication"""
        self.print_test_header("Cookie Authentication")
        
        try:
            # Test without Authorization header (should use cookie)
            response = self.session.get(f"{BASE_URL}/auth/me")
            
            if response.status_code == 200:
                user_info = response.json()
                return self.print_result(True, 
                    f"Cookie auth successful. User: {user_info['username']}")
            else:
                return self.print_result(False, f"Cookie auth failed: {response.status_code}")
                
        except Exception as e:
            return self.print_result(False, f"Cookie auth error: {str(e)}")
            
    def test_6_protected_route_access(self):
        """Test access to protected routes"""
        self.print_test_header("Protected Route Access")
        
        try:
            # Test root endpoint (should redirect to login if not authenticated)
            response = self.session.get(f"{BASE_URL}/")
            
            if response.status_code == 200:
                return self.print_result(True, "Successfully accessed protected route")
            elif response.status_code == 307:  # Redirect
                return self.print_result(True, "Redirected as expected (not authenticated)")
            else:
                return self.print_result(False, f"Unexpected status: {response.status_code}")
                
        except Exception as e:
            return self.print_result(False, f"Protected route error: {str(e)}")
            
    def test_7_authenticated_route_access(self):
        """Test access to protected routes when authenticated"""
        self.print_test_header("Authenticated Route Access")
        
        if not self.access_token:
            return self.print_result(False, "No access token available")
            
        try:
            # Test with valid token
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = self.session.get(f"{BASE_URL}/", headers=headers)
            
            if response.status_code == 200:
                return self.print_result(True, "Successfully accessed protected route with token")
            else:
                return self.print_result(False, f"Token access failed: {response.status_code}")
                
        except Exception as e:
            return self.print_result(False, f"Authenticated access error: {str(e)}")
            
    def test_8_user_logout(self):
        """Test user logout functionality"""
        self.print_test_header("User Logout")
        
        try:
            response = self.session.post(f"{BASE_URL}/auth/logout")
            
            if response.status_code == 200:
                # Check if cookie was cleared
                cookies = response.cookies
                auth_cookie_cleared = 'access_token' not in cookies or cookies['access_token'].value == ''
                
                return self.print_result(True, 
                    f"Logout successful. Cookie cleared: {auth_cookie_cleared}")
            else:
                return self.print_result(False, f"Logout failed: {response.status_code}")
                
        except Exception as e:
            return self.print_result(False, f"Logout error: {str(e)}")
            
    def test_9_post_logout_access(self):
        """Test access after logout"""
        self.print_test_header("Post-Logout Access")
        
        try:
            # Try to access protected route after logout
            response = self.session.get(f"{BASE_URL}/auth/me")
            
            if response.status_code == 401:
                return self.print_result(True, "Correctly denied access after logout")
            else:
                return self.print_result(False, f"Unexpected access after logout: {response.status_code}")
                
        except Exception as e:
            return self.print_result(False, f"Post-logout test error: {str(e)}")
            
    def test_10_environment_variables(self):
        """Test environment variable configuration"""
        self.print_test_header("Environment Variables")
        
        required_vars = [
            "MYSQL_HOST",
            "MYSQL_USER", 
            "MYSQL_PASSWORD",
            "MYSQL_DATABASE",
            "JWT_SECRET_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if not missing_vars:
            return self.print_result(True, "All required environment variables are set")
        else:
            return self.print_result(False, f"Missing environment variables: {', '.join(missing_vars)}")
            
    def run_all_tests(self):
        """Run all authentication tests"""
        print("🚀 STARTING COMPREHENSIVE AUTHENTICATION SYSTEM TEST")
        print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Testing against: {BASE_URL}")
        
        tests = [
            self.test_1_server_connectivity,
            self.test_2_user_registration,
            self.test_3_user_login,
            self.test_4_jwt_token_validation,
            self.test_5_cookie_authentication,
            self.test_6_protected_route_access,
            self.test_7_authenticated_route_access,
            self.test_8_user_logout,
            self.test_9_post_logout_access,
            self.test_10_environment_variables
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
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Authentication system is working correctly.")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
            
        return passed == total

def main():
    """Main test execution"""
    tester = AuthenticationSystemTester()
    
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
