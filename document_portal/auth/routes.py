"""
Authentication routes for login, registration, and user management
"""

from fastapi import APIRouter, HTTPException, status, Depends, Response
from fastapi.security import HTTPBearer
from fastapi.responses import RedirectResponse
from typing import Optional
from datetime import timedelta
from models.user import (
    UserCreate, UserLogin, UserResponse, user_manager, User
)
from auth.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """Register a new user"""
    try:
        user = user_manager.create_user(user_data)
        return UserResponse.model_validate(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login")
async def login(user_credentials: UserLogin, response: Response):
    """Login user and return access token"""
    user = user_manager.authenticate_user(
        user_credentials.username, 
        user_credentials.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=user_manager.access_token_expire_minutes)
    access_token = user_manager.create_access_token(
        data={"sub": str(user.id)}, 
        expires_delta=access_token_expires
    )
    
    # Set the token in a secure HTTP-only cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=user_manager.access_token_expire_minutes * 60
    )
    
    # Return success response with token (for frontend storage)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(user),
        "redirect_url": "/"
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse.model_validate(current_user)

@router.post("/logout")
async def logout(response: Response):
    """Logout user (client should discard token)"""
    # Clear the cookie
    response.delete_cookie(key="access_token")
    return {"message": "Successfully logged out"}

@router.post("/refresh")
async def refresh_token(current_user: User = Depends(get_current_user)):
    """Refresh access token"""
    access_token_expires = timedelta(minutes=user_manager.access_token_expire_minutes)
    access_token = user_manager.create_access_token(
        data={"sub": str(current_user.id)}, 
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/check")
async def check_auth(current_user: Optional[User] = Depends(get_current_user)):
    """Check if user is authenticated"""
    if current_user:
        return {
            "authenticated": True,
            "user": UserResponse.model_validate(current_user)
        }
    else:
        return {"authenticated": False}

