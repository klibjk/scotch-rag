#!/usr/bin/env python3
"""
Test script to understand FastHTML request handling
"""

import requests
import json

def test_api():
    """Test the API endpoints"""
    
    base_url = "http://localhost:8000"
    
    # Test status endpoint
    print("Testing status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Status test failed: {e}")
    
    # Test ask endpoint with different methods
    print("\nTesting ask endpoint...")
    
    # Method 1: JSON in body
    try:
        headers = {"Content-Type": "application/json"}
        data = {"question": "What products are in this dataset?"}
        response = requests.post(f"{base_url}/api/ask", json=data, headers=headers)
        print(f"Method 1 (JSON): {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Form data
    try:
        data = {"question": "What products are in this dataset?"}
        response = requests.post(f"{base_url}/api/ask", data=data)
        print(f"Method 2 (Form): {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Query parameters
    try:
        params = {"question": "What products are in this dataset?"}
        response = requests.post(f"{base_url}/api/ask", params=params)
        print(f"Method 3 (Query): {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Method 3 failed: {e}")

if __name__ == "__main__":
    test_api()
