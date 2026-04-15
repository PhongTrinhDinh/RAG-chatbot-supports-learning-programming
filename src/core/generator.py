from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import google.generativeai as genai
import logging
import hashlib
import json
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# ===========================================================
#           Base LLM
# ===========================================================

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

# ===========================================================
#                Gemini implement
# ===========================================================

class GeminiLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = ""):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        try:
            logger.info(f"Generating content with model: {self.model.model_name}")
            response = self.model.generate_content(prompt)
            logger.info("Successfully generated content")
            return response.text
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            # Return a user-friendly error message
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"

# ==============================================================
#           LLM factory
# =============================================================

class LLMFactory:
    @staticmethod
    def create(provider: str, **kwargs) -> BaseLLM:
        if provider == "gemini":
            return GeminiLM(**kwargs)
        raise ValueError(f"Unsupported LLM provider: {provider}")

# ==============================================================
#               Improved Reranker
# ==============================================================

class Reranker:
    def __init__(self, embedder=None):
        self.embedder = embedder

    def rerank(self, query: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Rerank retrieved documents based on relevance to query.
        Uses cosine similarity between query and document embeddings.
        """
        if not retrieved_docs:
            return retrieved_docs

        if self.embedder is None:
            logger.warning("No embedder provided for reranking, returning original order")
            return retrieved_docs

        try:
            # Embed the query
            query_embedding = self.embedder.embed_text(query)

            # Score each document
            scored_docs = []
            for doc in retrieved_docs:
                doc_embedding = self.embedder.embed_text(doc["document"])
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                scored_docs.append((similarity, doc))

            # Sort by similarity score (descending)
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # Return documents in new order
            reranked_docs = [doc for score, doc in scored_docs]
            logger.info(f"Reranked {len(retrieved_docs)} documents")
            return reranked_docs

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Fallback to original order if reranking fails
            return retrieved_docs

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

# =================================================================
#                  Improved Context builder
# ==============================================================

class ContextBuilder:
    def __init__(self, embedder=None, max_context_tokens: int = 4000):
        self.embedder = embedder
        self.max_context_tokens = max_context_tokens

    def build(self, retrieved_docs: List[Dict]) -> str:
        """
        Build context from retrieved documents, respecting token limits.
        """
        if not retrieved_docs:
            return ""

        context_parts = []
        current_tokens = 0

        for doc in retrieved_docs:
            chunk_text = doc["document"]

            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            # For more accurate counting, we'd need the tokenizer, but this is a reasonable estimate
            estimated_tokens = len(chunk_text) // 4

            if current_tokens + estimated_tokens > self.max_context_tokens:
                logger.info(f"Context limit reached: {current_tokens} tokens used, stopping at document {len(context_parts)+1}")
                break

            context_parts.append(chunk_text)
            current_tokens += estimated_tokens

        logger.info(f"Built context with {len(context_parts)} documents, ~{current_tokens} tokens")
        return "\n\n".join(context_parts)

# ================================================================
#              Main RAG generation
# =============================================================

class RAGGenerator:
    def __init__(self, embedder, vector_store, llm: BaseLLM,
                 reranker: Optional[Reranker] = None,
                 context_builder: Optional[ContextBuilder] = None,
                 prompt_template: Optional[str] = None,
                 enable_caching: bool = True):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.reranker = reranker or Reranker(embedder)
        self.context_builder = context_builder or ContextBuilder(embedder)
        self.enable_caching = enable_caching

        # Default prompt template
        self.prompt_template = prompt_template or """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not in context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        # Simple cache for query-response pairs
        self._cache = {} if enable_caching else None

    def answer(self, query: str, top_k: int = 5) -> Dict:
        # Check cache first
        if self.enable_caching:
            cache_key = self._get_cache_key(query, top_k)
            if cache_key in self._cache:
                logger.info(f"Returning cached response for query: {query[:50]}...")
                return self._cache[cache_key]

        logger.info(f"Processing query: {query}")

        # step 1: embed query
        query_vector = self.embedder.embed_text(query)

        # step 2: retrieve
        retrieved_docs = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k
        )

        # step 3: rerank
        reranker_docs = self.reranker.rerank(
            query=query,
            retrieved_docs=retrieved_docs
        )

        # step 4: build context
        context = self.context_builder.build(reranker_docs)

        # step 5: Prompt Construction
        prompt = self._build_prompt(
            query=query,
            context=context
        )

        # step 6: generate
        answer = self.generate(prompt)

        result = {
            "query": query,
            "answer": answer,
            "context": context,
            "sources": reranker_docs
        }

        # Cache the result
        if self.enable_caching:
            self._cache[cache_key] = result
            # Limit cache size to prevent memory issues
            if len(self._cache) > 100:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

        return result

    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate a cache key for the query and parameters."""
        key_data = {
            "query": query,
            "top_k": top_k
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    def _build_prompt(self, query: str, context: str) -> str:
        return self.prompt_template.format(
            query=query,
            context=context
        )

    def clear_cache(self):
        """Clear the response cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Response cache cleared")