#!/usr/bin/env python3
"""
Test the exact auth route logic
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_auth_route_logic():
    """Test the exact logic from the auth route"""
    
    try:
        print("🔍 Testing Auth Route Logic")
        print("=" * 50)
        
        # Import after adding to path
        from models.user import user_manager, UserLogin
        
        # Test credentials
        test_username = "Jeevankumar"
        test_password = input("Enter your password: ")
        
        print(f"\n🧪 Testing with username: {test_username}")
        
        # Step 1: Create UserLogin object (exactly like the route)
        print("\n1️⃣ Creating UserLogin object...")
        user_credentials = UserLogin(username=test_username, password=test_password)
        print(f"   ✅ UserLogin created: username={user_credentials.username}")
        
        # Step 2: Call authenticate_user (exactly like the route)
        print("\n2️⃣ Calling user_manager.authenticate_user...")
        user = user_manager.authenticate_user(
            user_credentials.username, 
            user_credentials.password
        )
        
        if user:
            print("✅ Authentication successful!")
            print(f"   User ID: {user.id}")
            print(f"   Username: {user.username}")
            print(f"   Email: {user.email}")
            
            # Step 3: Test JWT token creation (exactly like the route)
            print("\n3️⃣ Testing JWT token creation...")
            try:
                access_token_expires = timedelta(minutes=user_manager.access_token_expire_minutes)
                access_token = user_manager.create_access_token(
                    data={"sub": str(user.id)}, 
                    expires_delta=access_token_expires
                )
                print("✅ JWT token created successfully!")
                print(f"   Token preview: {access_token[:20]}...{access_token[-20:]}")
                
                # Step 4: Test UserResponse creation
                print("\n4️⃣ Testing UserResponse creation...")
                from models.user import UserResponse
                user_response = UserResponse.model_validate(user)
                print("✅ UserResponse created successfully!")
                print(f"   Response username: {user_response.username}")
                print(f"   Response email: {user_response.email}")
                
            except Exception as e:
                print(f"❌ JWT token creation failed: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print("❌ Authentication failed!")
            print("   This explains the 401 error in the API")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import timedelta
    test_auth_route_logic()
