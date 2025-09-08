"""
Fallback embedding options in case Jina AI API is unavailable
"""

from typing import List, Optional
from langchain_core.embeddings import Embeddings
import os

class OpenAIEmbeddings(Embeddings):
    """Fallback to OpenAI embeddings if Jina AI is unavailable"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for fallback embeddings")
        
        self.model = model
        self.base_url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        import requests
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return [item["embedding"] for item in result["data"]]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]


class HuggingFaceEmbeddings(Embeddings):
    """Fallback to Hugging Face Inference API if Jina AI is unavailable"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("Hugging Face API key is required for fallback embeddings")
        
        self.model = model
        self.base_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        import requests
        
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        return result if isinstance(result, list) else [result]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]


def get_embeddings_with_fallback():
    """Get embeddings with automatic fallback"""
    from jina_embeddings import JinaEmbeddings
    
    # Try Jina AI first
    try:
        print("🔄 Attempting to use Jina AI embeddings...")
        jina_api_key = os.getenv("JINA_AI_API_KEY")
        if jina_api_key:
            embeddings = JinaEmbeddings(api_key=jina_api_key)
            # Test with a simple query
            test_embedding = embeddings.embed_query("test")
            print("✅ Jina AI embeddings working!")
            return embeddings
    except Exception as e:
        print(f"❌ Jina AI failed: {e}")
    
    # Try OpenAI as fallback
    try:
        print("🔄 Attempting to use OpenAI embeddings as fallback...")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            test_embedding = embeddings.embed_query("test")
            print("✅ OpenAI embeddings working!")
            return embeddings
    except Exception as e:
        print(f"❌ OpenAI failed: {e}")
    
    # Try Hugging Face as last resort
    try:
        print("🔄 Attempting to use Hugging Face embeddings as fallback...")
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_api_key:
            embeddings = HuggingFaceEmbeddings(api_key=hf_api_key)
            test_embedding = embeddings.embed_query("test")
            print("✅ Hugging Face embeddings working!")
            return embeddings
    except Exception as e:
        print(f"❌ Hugging Face failed: {e}")
    
    raise Exception("All embedding services failed. Please check your API keys and network connection.")
