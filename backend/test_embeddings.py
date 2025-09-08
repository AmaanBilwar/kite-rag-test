#!/usr/bin/env python3
"""
Test script for Jina Embeddings v4 API integration
This script demonstrates how to use the Jina Embeddings v4 API with LangChain
"""

from dotenv import load_dotenv
from jina_embeddings import JinaEmbeddings
import os

load_dotenv()

def test_jina_embeddings():
    """Test the Jina Embeddings v4 API"""
    print("🚀 Testing Jina Embeddings v4 API")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("JINA_AI_API_KEY")
    if not api_key:
        print("❌ JINA_AI_API_KEY not found in environment variables")
        print("Please set your Jina AI API key:")
        print("1. Get your API key from https://jina.ai/")
        print("2. Add JINA_AI_API_KEY=your_key_here to your .env file")
        return
    
    # Test API key first with a simple request
    print("🔑 Testing API key...")
    import requests
    
    test_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    test_payload = {
        "model": "jina-embeddings-v4",
        "task": "text-matching",
        "input": [{"text": "test"}]
    }
    
    try:
        test_response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=test_headers,
            json=test_payload,
            timeout=60  # Increased timeout
        )
        print(f"API Key test status: {test_response.status_code}")
        if test_response.status_code != 200:
            print(f"API Key test error: {test_response.text}")
            return
        else:
            print("✅ API key is valid!")
    except requests.exceptions.Timeout as e:
        print(f"❌ API key test timed out: {e}")
        print("💡 The Jina AI API might be slow. Try again in a moment.")
        return
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return
    
    # Initialize the embedding model
    embeddings = JinaEmbeddings(
        api_key=api_key,
        model="jina-embeddings-v4",
        task="text-matching"
    )
    
    # Test single query embedding
    print("📝 Testing single query embedding...")
    query = "How does the authentication system work?"
    query_embedding = embeddings.embed_query(query)
    print(f"✅ Query embedding generated: {len(query_embedding)} dimensions")
    print(f"📊 Sample values: {query_embedding[:5]}")
    
    # Test multiple document embeddings
    print("\n📚 Testing multiple document embeddings...")
    documents = [
        "The authentication system uses JWT tokens for secure user sessions.",
        "Database queries are optimized with proper indexing for better performance.",
        "The API endpoints follow RESTful conventions for better maintainability.",
        "Error handling is implemented with proper logging and user feedback.",
    ]
    
    doc_embeddings = embeddings.embed_documents(documents)
    print(f"✅ Document embeddings generated: {len(doc_embeddings)} documents")
    print(f"📊 Each embedding has {len(doc_embeddings[0])} dimensions")
    
    # Test similarity calculation
    print("\n🔍 Testing similarity calculation...")
    import numpy as np
    
    # Calculate cosine similarity between query and documents
    query_norm = np.linalg.norm(query_embedding)
    similarities = []
    
    for i, doc_embedding in enumerate(doc_embeddings):
        doc_norm = np.linalg.norm(doc_embedding)
        similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
        similarities.append((i, similarity, documents[i]))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("📈 Document similarity rankings:")
    for rank, (doc_idx, similarity, doc_text) in enumerate(similarities, 1):
        print(f"{rank}. Similarity: {similarity:.4f} - {doc_text[:60]}...")
    
    print("\n✅ All tests completed successfully!")
    print("🎉 Jina Embeddings v4 API is working correctly!")
    print("💡 No local model download required - using cloud API!")

if __name__ == "__main__":
    test_jina_embeddings()
