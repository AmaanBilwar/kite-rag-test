#!/usr/bin/env python3
"""
Simple connectivity test for Jina AI API
"""

import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

def test_connectivity():
    """Test basic connectivity to Jina AI API"""
    print("🔍 Testing connectivity to Jina AI API...")
    
    # Test basic connectivity first
    try:
        print("1. Testing basic connectivity...")
        response = requests.get("https://api.jina.ai", timeout=10)
        print(f"   ✅ Basic connectivity: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Basic connectivity failed: {e}")
        return False
    
    # Test API endpoint
    try:
        print("2. Testing API endpoint...")
        response = requests.get("https://api.jina.ai/v1/embeddings", timeout=10)
        print(f"   ✅ API endpoint accessible: {response.status_code}")
    except Exception as e:
        print(f"   ❌ API endpoint failed: {e}")
        return False
    
    # Test with API key
    api_key = os.getenv("JINA_AI_API_KEY")
    if not api_key:
        print("   ⚠️  No API key found in environment")
        return False
    
    try:
        print("3. Testing with API key...")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        # Very simple payload
        payload = {
            "model": "jina-embeddings-v4",
            "task": "text-matching",
            "input": [{"text": "test"}]
        }
        
        start_time = time.time()
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )
        end_time = time.time()
        
        print(f"   ✅ API call successful: {response.status_code}")
        print(f"   ⏱️  Response time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   📊 Response structure: {list(result.keys())}")
            return True
        else:
            print(f"   ❌ API error: {response.text}")
            return False
            
    except requests.exceptions.Timeout as e:
        print(f"   ❌ API call timed out: {e}")
        return False
    except Exception as e:
        print(f"   ❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    success = test_connectivity()
    if success:
        print("\n🎉 All connectivity tests passed!")
    else:
        print("\n❌ Connectivity tests failed. Check your network and API key.")
