from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# ==================================================================================================
#                   Base Interface
# ==================================================================================================

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass
    
# =========================================================================
#                 Sentence Transformers Implement
# =========================================================================

class SentenceTransformersEmbedding(BaseEmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        
    def embed(self, text: str) -> List[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vectors.tolist()
    
# =======================================================================
#                   Factory
# =======================================================================
    
class EmbeddingFactory:
    @staticmethod
    def create(provider: str = "sentence_transformers", **kwargs) -> BaseEmbeddingModel:
        if provider == "sentence_transformers":
            return SentenceTransformersEmbedding(**kwargs)
            
        raise ValueError(f"Unsupported embedding provider: {provider}")
    
# ===========================================================================
#                            Processor
# ===========================================================================

class EmbeddingProcessor:
    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model
        
    def embed_text(self, text: str) -> List[float]:
        return self.embedding_model.embed(text)
    
    def embed_chunk(self, chunks: List[Any]) -> List[Dict]:
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_model.embed_batch(texts)
        
        embedded_chunks = []
        for chunk, vector in zip(chunks, vectors):
            embedded_chunks.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": vector,
                "metadata": chunk.metadata
            })
            
        return embedded_chunks