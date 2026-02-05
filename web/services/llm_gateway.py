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

# Config resolution (shared with autonomous_loop)
def _get_config_value(key: str, default: str = None) -> Optional[str]:
    val = os.environ.get(key)
    if val:
        return val
    env_path = "C:/shadow/backend/.env"
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith(f"{key}="):
                        return line.split("=", 1)[1].strip()
        except Exception:
            pass
    return default


class LLMGateway:
    """
    Unified gateway for all LLM interactions within the Python bridge.
    Tries Ollama first, then falls back to cloud APIs (Gemini, Claude, OpenAI).
    """
    def __init__(self):
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        self.default_model = os.environ.get("DEFAULT_MODEL", "glm4:9b")

        # Cloud API keys (loaded lazily)
        self._google_api_key = _get_config_value("GOOGLE_API_KEY")
        self._anthropic_api_key = _get_config_value("ANTHROPIC_API_KEY")
        self._openai_api_key = _get_config_value("OPENAI_API_KEY")

    async def ask(self, prompt: str, system_prompt: Optional[str] = None,
                  model: Optional[str] = None, json_mode: bool = False) -> str:
        """
        Ask the LLM a question and get a string response.
        Fallback chain: Ollama -> Gemini -> Claude -> OpenAI
        """
        model = model or self.default_model
        logger.info(f"LLM Gateway: Asking {model}...")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 1. Try Ollama (local)
        result = await self._try_ollama(messages, model, json_mode)
        if result is not None:
            return result

        # 2. Fallback to cloud APIs
        logger.warning("Ollama unavailable, trying cloud fallback chain...")

        # Gemini
        if self._google_api_key:
            result = await self._try_gemini(prompt, system_prompt)
            if result is not None:
                return result

        # Claude
        if self._anthropic_api_key:
            result = await self._try_claude(prompt, system_prompt)
            if result is not None:
                return result

        # OpenAI
        if self._openai_api_key:
            result = await self._try_openai(messages)
            if result is not None:
                return result

        return "Error: All LLM backends unavailable (Ollama, Gemini, Claude, OpenAI)"

    async def _try_ollama(self, messages: List[Dict], model: str,
                          json_mode: bool) -> Optional[str]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        if json_mode:
            payload["format"] = "json"

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.ollama_url}/api/chat", json=payload, timeout=60
                )
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                logger.warning(f"Ollama error: {response.status_code}")
                return None
        except Exception as e:
            logger.warning(f"Ollama unavailable: {e}")
            return None

    async def _try_gemini(self, prompt: str,
                          system_prompt: Optional[str]) -> Optional[str]:
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                "gemini-2.0-flash:generateContent"
                f"?key={self._google_api_key}"
            )
            parts = []
            if system_prompt:
                parts.append({"text": f"System: {system_prompt}\n\n{prompt}"})
            else:
                parts.append({"text": prompt})

            payload = {"contents": [{"parts": parts}]}
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, json=payload, timeout=60)
            )
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        logger.info("LLM Gateway: Gemini fallback succeeded")
                        return parts[0].get("text", "")
            logger.warning(f"Gemini error: {response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Gemini fallback failed: {e}")
            return None

    async def _try_claude(self, prompt: str,
                          system_prompt: Optional[str]) -> Optional[str]:
        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self._anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                payload["system"] = system_prompt

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    url, json=payload, headers=headers, timeout=90
                )
            )
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", [])
                if content:
                    logger.info("LLM Gateway: Claude fallback succeeded")
                    return content[0].get("text", "")
            logger.warning(f"Claude error: {response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Claude fallback failed: {e}")
            return None

    async def _try_openai(self, messages: List[Dict]) -> Optional[str]:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self._openai_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.7,
            }
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    url, json=payload, headers=headers, timeout=60
                )
            )
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    logger.info("LLM Gateway: OpenAI fallback succeeded")
                    return choices[0].get("message", {}).get("content", "")
            logger.warning(f"OpenAI error: {response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"OpenAI fallback failed: {e}")
            return None

    async def ask_json(self, prompt: str, system_prompt: Optional[str] = None,
                       model: Optional[str] = None) -> Dict[str, Any]:
        """Ask the LLM and expect a JSON response."""
        response_text = await self.ask(prompt, system_prompt, model, json_mode=True)
        try:
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
