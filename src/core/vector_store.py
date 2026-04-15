from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import chromadb
import logging as logger

# ==========================================================================
#                          Base Interface
# ==========================================================================

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, embedded_chunks: List[Dict]) -> None:
        pass
    
    @abstractmethod
    def similarity_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        pass
    
# ========================================================================
#                        ChromaDB implement
# ========================================================================

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chromadb"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def add_documents(self, embedded_chunks: List[Dict]) -> None:
        ids = [chunk["chunk_id"] for chunk in embedded_chunks]
        documents = [chunk["text"] for chunk in embedded_chunks]
        embeddings = [chunk["embedding"] for chunk in embedded_chunks]
        metadatas = [chunk["metadata"] for chunk in embedded_chunks]
        
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def similarity_search(self, query_vector: List[Dict], top_k: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        retrieved = []
        
        if not query_vector:
            raise ValueError("Query vector cannot be empty")
        
        if top_k <- 0:
            raise ValueError("top_k must be positive integer")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            
            retrieved = []
            for i in range(len(results['ids'][0])):
                retrieved.append({
                    "id": results['ids'][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
                
            #logger.info(f"Similarity search completed: {len(retrieved)} results returned")
            return retrieved
        
        except Exception as e:
            #logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)
        
# ==================================================================
#                             Factory layer
# ==================================================================

class VectorStoreFactory:
    @staticmethod
    def create(provider: str = "chroma", **kwargs) -> BaseVectorStore:
        if provider == "chroma":
            return ChromaVectorStore(**kwargs)
        raise ValueError(f"Unsupported vector store provider: {provider}")
    
# ===================================================================
#               Retrieval layer
# ===================================================================

class RetrievalProcessor:
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store
        
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        return self.vector_store.similarity_search(query_vector=query_vector, top_k=top_k)