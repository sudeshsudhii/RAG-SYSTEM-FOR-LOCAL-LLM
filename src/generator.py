"""
LLM generation module.

Provides an abstract LLM provider interface and multiple implementations
(Gemini, OpenAI, Ollama). The RAGGenerator class orchestrates the full
pipeline including query classification and observability tracking.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.prompts import (
    SYSTEM_PROMPT,
    NO_CONTEXT_RESPONSE,
    format_context,
    build_qa_prompt,
    build_rewrite_prompt,
)
from src.retriever import Retriever
from src.chunking import Chunk
from src.vector_store import SearchResult
from src.utils import setup_logger
from src.classifier import QueryClassifier
from src.observability import rag_logger

logger = setup_logger(__name__)


# ── Response Dataclass ───────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieved_chunks: list[SearchResult] = field(default_factory=list)
    confidence: float = 0.0
    query_used: str = ""
    tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    cost: float = 0.0


# ── LLM Provider Interface ──────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a response from the LLM.
        Returns the generated text.
        """
        ...
        
    @abstractmethod
    def generate_with_metrics(self, prompt: str, system_prompt: str = "") -> Tuple[str, dict]:
        """
        Generate a response and return token metrics.
        Returns (generated_text, token_usage_dict).
        """
        ...


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash", temperature: float = 0.1, max_tokens: int = 2048):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai is required.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_PROMPT)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.provider_name = "gemini"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        text, _ = self.generate_with_metrics(prompt, system_prompt)
        return text

    def generate_with_metrics(self, prompt: str, system_prompt: str = "") -> Tuple[str, dict]:
        import google.generativeai as genai
        try:
            model = self.model
            if system_prompt:
                model = genai.GenerativeModel(model_name=self.model_name, system_instruction=system_prompt)
                
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            text = response.text.strip() if response.text else NO_CONTEXT_RESPONSE
            
            # Approximate token counts if metadata isn't easily accessible
            # Assuming ~4 chars per token for English
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(text) // 4
            
            metrics = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            return text, metrics

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.1, max_tokens: int = 2048):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required.")

        self.client = OpenAI(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.provider_name = "openai"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        text, _ = self.generate_with_metrics(prompt, system_prompt)
        return text

    def generate_with_metrics(self, prompt: str, system_prompt: str = "") -> Tuple[str, dict]:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
                
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            text = response.choices[0].message.content.strip()
            usage = response.usage
            
            metrics = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
            return text, metrics
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model_name: str = "llama3", temperature: float = 0.1, max_tokens: int = 2048):
        try:
            import requests
        except ImportError:
            raise ImportError("requests package is required for Ollama.")

        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.provider_name = "ollama"

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        text, _ = self.generate_with_metrics(prompt, system_prompt)
        return text

    def generate_with_metrics(self, prompt: str, system_prompt: str = "") -> Tuple[str, dict]:
        import requests
        try:
            sys_p = system_prompt if system_prompt else SYSTEM_PROMPT
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": sys_p,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            
            data = response.json()
            text = data.get("response", "").strip()
            
            metrics = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            }
            return text, metrics
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e


class LLMFactory:
    """Factory to create LLM providers based on config."""
    @staticmethod
    def create(config) -> LLMProvider:
        provider = config.llm_provider.lower()
        if provider == "openai":
            return OpenAIProvider(
                api_key=config.openai_api_key,
                model_name=config.openai_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
        elif provider == "ollama":
            return OllamaProvider(
                base_url=config.ollama_base_url,
                model_name=config.ollama_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
        else:
            return GeminiProvider(
                api_key=config.gemini_api_key,
                model_name=config.gemini_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )


# ── RAG Generator ────────────────────────────────────────────────────────────

class RAGGenerator:
    """
    Orchestrates the full RAG pipeline: retrieve → build context → generate.
    """
    def __init__(self, llm: LLMProvider, retriever: Retriever, config) -> None:
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.classifier = QueryClassifier(self.llm)

    def answer(
        self,
        query: str,
        query_id: str = "query_default",
        top_k: int = 5,
        enable_hybrid: bool = False,
        enable_reranking: bool = False,
        enable_query_rewrite: bool = False,
        chunks: Optional[list[Chunk]] = None,
    ) -> RAGResponse:
        
        start_time = time.time()
        query_used = query
        
        # 1. Classification
        category = self.classifier.classify(query)
        rag_logger.log_query(query_id, query, category=category)
        
        if category == "Greeting":
            return RAGResponse(answer="Hello! I am ready to answer questions based on your uploaded documents. What would you like to know?")
        elif category == "Small Talk":
            return RAGResponse(answer="I'm doing well! But I'm specifically designed to help you search and understand the documents you've uploaded. How can I assist you with those?")
        elif category == "Unsupported":
            return RAGResponse(answer="I can only assist with answering questions based on the provided documents.")

        # 2. Query Rewriting
        if enable_query_rewrite:
            try:
                rewrite_prompt = build_rewrite_prompt(query)
                rewritten = self.llm.generate(rewrite_prompt, system_prompt="You are a strict query optimizer.")
                if rewritten and rewritten != query:
                    logger.info(f"Query rewritten: '{query[:50]}...' → '{rewritten[:50]}...'")
                    query_used = rewritten
                    rag_logger.log_query(query_id, query, rewritten_query=query_used, category=category)
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}")

        # 3. Retrieval
        retrieval_start = time.time()
        if enable_hybrid and chunks:
            results = self.retriever.hybrid_retrieve(query_used, chunks, top_k=top_k * 2 if enable_reranking else top_k)
        else:
            results = self.retriever.retrieve(query_used, top_k=top_k * 2 if enable_reranking else top_k)

        if enable_reranking and results:
            results = self.retriever.rerank(query_used, results, top_k=top_k)
        elif len(results) > top_k:
            results = results[:top_k]
            
        retrieval_latency = (time.time() - retrieval_start) * 1000

        # Handle empty retrieval
        if not results:
            rag_logger.log_retrieval(query_id, 0, 0.0, retrieval_latency)
            return RAGResponse(answer=NO_CONTEXT_RESPONSE, query_used=query_used)

        confidence = self._calculate_confidence(results)
        rag_logger.log_retrieval(query_id, len(results), confidence, retrieval_latency)

        # 4. Context & Generation
        gen_start = time.time()
        context = format_context(results)
        prompt = build_qa_prompt(context, query)

        try:
            answer, metrics = self.llm.generate_with_metrics(prompt)
            gen_latency = (time.time() - gen_start) * 1000
            
            cost = self._calculate_cost(metrics)
            
            rag_logger.log_generation(
                query_id, 
                metrics["prompt_tokens"], 
                metrics["completion_tokens"], 
                metrics["total_tokens"], 
                cost, 
                gen_latency,
                getattr(self.llm, "provider_name", "unknown"),
                getattr(self.llm, "model_name", "unknown")
            )
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            rag_logger.log_error(query_id, str(e), "generator")
            return RAGResponse(answer=f"Error generating answer: {e}", retrieved_chunks=results, query_used=query_used)

        sources = self._extract_sources(results)

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=results,
            confidence=confidence,
            query_used=query_used,
            tokens=metrics,
            cost=cost
        )

    def _extract_sources(self, results: list[SearchResult]) -> list[dict]:
        seen = set()
        sources = []
        for result in results:
            doc = result.metadata.get("document", "Unknown")
            page = result.metadata.get("page", "N/A")
            key = (doc, page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": doc,
                    "page": page,
                    "chunk_id": result.metadata.get("chunk_id", "N/A"),
                    "score": result.score,
                })
        return sources

    def _calculate_confidence(self, results: list[SearchResult]) -> float:
        if not results: return 0.0
        return max(0.0, min(1.0, sum(r.score for r in results) / len(results)))
        
    def _calculate_cost(self, metrics: dict) -> float:
        if not self.config: return 0.0
        in_cost = (metrics.get("prompt_tokens", 0) / 1_000_000) * self.config.cost_per_1m_input
        out_cost = (metrics.get("completion_tokens", 0) / 1_000_000) * self.config.cost_per_1m_output
        return in_cost + out_cost
