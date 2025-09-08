# Jina Embeddings v4 Setup Guide

This guide explains how to use the Jina Embeddings v4 model in your RAG system.

## Overview

Jina Embeddings v4 is a universal embedding model for multimodal and multilingual retrieval, built on Qwen2.5-VL-3B-Instruct. It features:

- **Unified embeddings** for text, images, and visual documents
- **Multilingual support** (30+ languages)
- **Task-specific adapters** for retrieval, text matching, and code-related tasks
- **Flexible embedding size**: 2048 dimensions by default, can be truncated to 128

## Installation

The required dependencies are already included in `pyproject.toml`:

```toml
dependencies = [
    "transformers>=4.52.0",
    "torch>=2.6.0", 
    "peft>=0.15.2",
    "torchvision",
    "pillow",
    "langchain-huggingface>=0.3.1",
    # ... other dependencies
]
```

Install dependencies:
```bash
cd backend
uv sync
```

## Configuration

The embedding model is configured in `main.py` with optimal settings:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v4",
    model_kwargs={
        "trust_remote_code": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    encode_kwargs={
        "normalize_embeddings": True,  # Normalize embeddings for better similarity search
        "batch_size": 32,  # Process embeddings in batches for efficiency
        "show_progress_bar": True,  # Show progress during embedding generation
    }
)
```

## Key Features

### 1. Task-Specific Configuration
- **Retrieval Task**: Optimized for document retrieval and similarity search
- **Text Matching**: For semantic similarity between texts
- **Code Understanding**: For code-related tasks

### 2. Multilingual Support
The model supports 30+ languages including:
- English, Chinese, Arabic, French, German, Spanish
- Japanese, Korean, Hindi, Italian, Portuguese, and more

### 3. Performance Optimizations
- **Normalized embeddings**: Better similarity search results
- **Batch processing**: Efficient embedding generation
- **GPU acceleration**: Automatic CUDA detection and usage
- **Progress tracking**: Visual feedback during processing

## Usage Examples

### Basic Usage
```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v4",
    model_kwargs={"trust_remote_code": True}
)

# Generate embeddings
query_embedding = embeddings.embed_query("What is machine learning?")
doc_embeddings = embeddings.embed_documents(["Machine learning is...", "AI involves..."])
```

### Advanced Configuration
```python
embeddings = HuggingFaceEmbeddings(
    model_name="jinaai/jina-embeddings-v4",
    model_kwargs={
        "trust_remote_code": True,
        "device": "cuda",  # Force GPU usage
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 64,
        "show_progress_bar": True,
    }
)
```

## Testing

Run the test script to verify the setup:

```bash
cd backend
python test_embeddings.py
```

This will:
1. Load the Jina Embeddings v4 model
2. Generate embeddings for sample queries and documents
3. Calculate similarity scores
4. Display the results

## Performance Tips

1. **Normalize embeddings**: Enable `normalize_embeddings=True` for better similarity
2. **Batch processing**: Use `batch_size` for multiple documents
3. **GPU acceleration**: Set `device="cuda"` to use GPU if available
4. **Progress tracking**: Enable `show_progress_bar=True` for visual feedback
5. **Memory management**: Reduce `batch_size` if you encounter memory issues

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure `trust_remote_code=True` is set
2. **CUDA out of memory**: Reduce `batch_size` or use CPU
3. **Slow performance**: Enable GPU acceleration or reduce embedding dimensions

### Memory Requirements
- **CPU**: ~8GB RAM recommended
- **GPU**: ~4GB VRAM recommended for optimal performance

## References

- [Jina Embeddings v4 on Hugging Face](https://huggingface.co/jinaai/jina-embeddings-v4)
- [LangChain HuggingFace Integration](https://python.langchain.com/docs/integrations/text_embedding/huggingface)
- [Technical Report](https://arxiv.org/abs/2506.18902)
