from dotenv import load_dotenv
from jina_embeddings import JinaEmbeddings
import os
from langchain.chat_models import init_chat_model
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from typing import List, TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

load_dotenv()
CHAT_MODEL = "openai/gpt-oss-120b"
groq_api_key = os.getenv("GROQ_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
jina_api_key = os.getenv("JINA_AI_API_KEY")

if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("LANGSMITH_API_KEY is not set")
if not jina_api_key:
    raise ValueError("JINA_AI_API_KEY is not set. Get your API key from https://jina.ai/")



llm = init_chat_model(CHAT_MODEL, model_provider="groq", api_key=groq_api_key)

# Configure Jina Embeddings v4 using API
embeddings = JinaEmbeddings(
    api_key=jina_api_key,
    model="jina-embeddings-v4",
    task="text-matching"  # Use text-matching for general purpose embeddings
)

# Test the embedding model
print("Testing Jina Embeddings v4 API...")
try:
    test_embedding = embeddings.embed_query("Hello, world!")
    print(f"✅ Jina AI API connected successfully!")
    print(f"📊 Embedding dimension: {len(test_embedding)}")
    print(f"🔢 Sample embedding values: {test_embedding[:5]}...")
except Exception as e:
    print(f"❌ Error connecting to Jina AI API: {e}")
    raise

# Define the directory containing your code files
# You can add multiple directories or specific file paths
documents = "C:/Users/amaan/OneDrive/Documents/coding/kite-rag/backend/"  

embedding_dim = len(test_embedding)
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Simple document loader that doesn't require tree-sitter
loader = DirectoryLoader(
    documents,
    glob="**/*",
    show_progress=False,
    use_multithreading=False
)

try:
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from the codebase")
except Exception as e:
    print(f"Error loading documents: {e}")
    docs = []

# Simple text splitter that works without tree-sitter
def split_documents(docs):
    """Split documents using a simple text splitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_splits = []
    for doc in docs:
        splits = text_splitter.split_documents([doc])
        all_splits.extend(splits)
    
    return all_splits

all_splits = split_documents(docs)
print(f"Created {len(all_splits)} text chunks from documents")

# Index chunks
if all_splits:
    try:
        _ = vector_store.add_documents(documents=all_splits)
        print(f"Successfully indexed {len(all_splits)} chunks in vector store")
    except Exception as e:
        print(f"Error indexing documents: {e}")
else:
    print("No documents to index")

# Create proper prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You're Kite, an AI assistant that can answer questions about the codebase. 
    You are given a question and a context containing code from multiple programming languages.
    You need to answer the question based on the context. 
    If you don't know the answer, you should say you don't know. 
    You should also cite the source of the answer when possible.
    
    The codebase may contain files in various programming languages including:
    - Python (.py)
    - JavaScript (.js, .jsx)
    - TypeScript (.ts, .tsx)
    - Markdown (.md)
    - HTML (.html)
    - Solidity (.sol)
    - C# (.cs)
    - Haskell (.hs)
    - PHP (.php)
    - PowerShell (.ps1)
    - Visual Basic (.vb)
    
    Provide clear, accurate answers based on the code context provided."""),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    # Use query-specific embedding for better retrieval
    query_embedding = embeddings.embed_query(state["question"])
    retrieved_docs = vector_store.similarity_search_by_vector(query_embedding, k=5)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()




# Test the RAG system
if len(all_splits) > 0:
    print("\n" + "="*50)
    print("Testing the RAG system...")
    print("="*50)
    
    test_questions = [
        "what languge is the code in main.py written in?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            response = graph.invoke({"question": question})
            print(f"Answer: {response['answer']}")
        except Exception as e:
            print(f"Error processing question: {e}")
        print("-" * 30)
else:
    print("No documents were loaded. Please check your file paths and ensure code files exist.")