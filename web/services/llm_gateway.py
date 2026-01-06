"""
LLM Gateway - AGI-Readiness Infrastructure

Provides a unified interface for Python services to interact with LLM backends:
1. Local Ollama (GLM-4, Llama 3)
2. Cloud APIs (Anthropic, Gemini, OpenAI) via HTTP
3. Smart fallback and routing
"""
import logging
import json
import os
import requests
import asyncio
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class LLMGateway:
    """
    Unified gateway for all LLM interactions within the Python bridge.
    """
    def __init__(self):
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        self.default_model = os.environ.get("DEFAULT_MODEL", "glm4:9b")
        
    async def ask(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Ask the LLM a question and get a string response.
        Currently defaults to Ollama, but can be extended to cloud APIs.
        """
        model = model or self.default_model
        logger.info(f"LLM Gateway: Asking {model}...")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }
        
        if json_mode:
            payload["format"] = "json"

        try:
            # Run in executor to not block async loop if requests is used
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                logger.error(f"LLM Gateway Error: {response.status_code} - {response.text}")
                return f"Error: LLM returned status {response.status_code}"
                
        except Exception as e:
            logger.error(f"LLM Gateway Exception: {e}")
            return f"Exception: {str(e)}"

    async def ask_json(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Ask the LLM and expect a JSON response."""
        response_text = await self.ask(prompt, system_prompt, model, json_mode=True)
        try:
            # Clean response text in case of markdown blocks
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON response: {e}\nResponse: {response_text}")
            return {"error": "JSON parse error", "raw": response_text}

# Global instance
_gateway: Optional[LLMGateway] = None

def get_llm_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway
