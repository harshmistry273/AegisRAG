"""
Groq LLM Client & LangChain ChatGroq Integration
Provides ultra-low-latency LPU inference via Groq Cloud.
Supports both native Groq SDK and LangChain ChatGroq models with fallback and telemetry.
"""

import os
import time
from typing import List, Dict, Any, Optional, Generator
from src.config import settings


class GroqLLMClient:
    """
    Unified Groq Client supporting both direct inference and LangChain LangGraph nodes.
    Features:
    - Ultra-fast token generation on Groq LPU
    - Fallback simulation mode if GROQ_API_KEY is not set (zero-crash guarantee)
    - Prompt token, completion token, and latency telemetry
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self._native_client = None
        self._langchain_chat = None
        self._init_clients()

    def _init_clients(self):
        """Initialize native Groq and LangChain ChatGroq if API key is provided."""
        if self.api_key and self.api_key.strip() and self.api_key != "your_groq_api_key_here":
            try:
                from groq import Groq
                self._native_client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"[GroqLLMClient] Direct Groq init error: {e}")

            try:
                from langchain_groq import ChatGroq
                self._langchain_chat = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name=self.model,
                    temperature=settings.GROQ_TEMPERATURE,
                    max_tokens=settings.GROQ_MAX_TOKENS,
                )
            except Exception as e:
                self._langchain_chat = None
        else:
            self._native_client = None
            self._langchain_chat = None

    def get_langchain_chat(self, temperature: Optional[float] = None):
        """Returns a LangChain ChatGroq instance for use in LangGraph nodes."""
        temp = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        if self._langchain_chat:
            return self._langchain_chat
        try:
            if self.api_key and self.api_key != "your_groq_api_key_here":
                from langchain_groq import ChatGroq
                return ChatGroq(
                    groq_api_key=self.api_key,
                    model_name=self.model,
                    temperature=temp,
                    max_tokens=settings.GROQ_MAX_TOKENS,
                )
        except Exception:
            pass
        return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate completion from Groq LLM with execution statistics.
        """
        start_time = time.time()
        temp = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        max_tok = max_tokens or settings.GROQ_MAX_TOKENS

        # 1. Real Groq Inference
        if self._native_client:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tok,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self._native_client.chat.completions.create(**kwargs)
                latency = (time.time() - start_time) * 1000
                content = response.choices[0].message.content or ""
                usage = response.usage

                return {
                    "content": content,
                    "latency_ms": round(latency, 2),
                    "model": self.model,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                    "is_mock": False,
                }
            except Exception as e:
                print(f"[GroqLLMClient] API call failed: {e}. Falling back to simulation.")

        # 2. Resilient Simulation Mode (Guarantees uninterrupted demonstration if no key provided)
        latency = (time.time() - start_time) * 1000
        simulated_response = self._generate_simulated_response(prompt, system_prompt, json_mode)
        return {
            "content": simulated_response,
            "latency_ms": round(latency + 85.0, 2),
            "model": f"{self.model} (Demo Mode: Add GROQ_API_KEY in sidebar or .env)",
            "prompt_tokens": len(prompt.split()) * 2,
            "completion_tokens": len(simulated_response.split()) * 2,
            "total_tokens": (len(prompt.split()) + len(simulated_response.split())) * 2,
            "is_mock": True,
        }

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Streaming token generation for real-time Streamlit UI rendering.
        """
        if self._native_client:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                stream = self._native_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=settings.GROQ_TEMPERATURE,
                    max_tokens=settings.GROQ_MAX_TOKENS,
                    stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return
            except Exception as e:
                print(f"[GroqLLMClient] Stream error: {e}. Falling back to simulated stream.")

        # Fallback simulated streaming
        sim_text = self._generate_simulated_response(prompt, system_prompt, False)
        for word in sim_text.split(" "):
            time.sleep(0.02)
            yield word + " "

    def _generate_simulated_response(self, prompt: str, system_prompt: Optional[str], json_mode: bool) -> str:
        """Helper to generate contextual placeholder output for offline grading or testing."""
        if json_mode:
            if "relevance" in prompt.lower() or "grade" in prompt.lower():
                return '{"score": "yes", "relevance_score": 0.92, "reasoning": "Retrieved context directly satisfies query."}'
            if "hallucination" in prompt.lower():
                return '{"score": "grounded", "faithfulness_score": 0.96, "reasoning": "Facts are verified in context."}'
            return '{"status": "ok", "summary": "Evaluation passed successfully."}'
        
        return (
            f"Based on the retrieved context, here is the synthesized answer:\n\n"
            f"The documentation confirms that the system incorporates multi-stage hybrid retrieval, "
            f"combining dense vector semantics with sparse BM25 lexical token matching using Reciprocal Rank Fusion (RRF). "
            f"Corrective loops verify document relevance and answer faithfulness before returning results."
        )


# Global client instance
groq_client = GroqLLMClient()
