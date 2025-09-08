"""
Jina AI Embeddings API Integration
Custom implementation to use Jina AI's API endpoint directly
"""

import os
import requests
import time
from typing import List, Optional, Dict, Any
from langchain_core.embeddings import Embeddings
import json


class JinaEmbeddings(Embeddings):
    """Jina AI Embeddings using their API endpoint"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-embeddings-v4",
        task: str = "text-matching",
        base_url: str = "https://api.jina.ai/v1/embeddings",
        **kwargs: Any,
    ):
        """
        Initialize Jina AI Embeddings
        
        Args:
            api_key: Jina AI API key. If not provided, will use JINA_AI_API_KEY env var
            model: Model name (default: jina-embeddings-v4)
            task: Task type - "text-matching", "retrieval.query", "retrieval.passage", "code.query", or "code.passage"
            base_url: API base URL
        """
        self.api_key = api_key or os.getenv("JINA_AI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina AI API key is required. Set JINA_AI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model = model
        self.task = task
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    def _make_request(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """Make API request to Jina AI with retry logic"""
        
        # Prepare input data - Jina AI expects array of objects with "text" key
        input_data = [{"text": text} for text in texts]
        
        payload = {
            "model": self.model,
            "task": self.task,
            "input": input_data
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract embeddings from response
                embeddings = []
                for item in result.get("data", []):
                    embeddings.append(item.get("embedding", []))
                
                return embeddings
                
            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⏳ Timeout on attempt {attempt + 1}, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Timeout calling Jina AI API after {max_retries} attempts: {e}. The API might be slow or overloaded.")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"🔄 Request failed on attempt {attempt + 1}, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Error calling Jina AI API after {max_retries} attempts: {e}")
            except json.JSONDecodeError as e:
                raise Exception(f"Error parsing API response: {e}")
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in retry logic")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        return self._make_request(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self._make_request([text])
        return embeddings[0] if embeddings else []
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        return self._make_request(texts)
