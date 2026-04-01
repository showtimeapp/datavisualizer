"""
Unified LLM client — supports OpenAI and Gemini via their REST APIs.
No SDK dependency, just httpx. Includes retry logic for rate limits.
"""

import httpx
import json
import asyncio
import logging

from app.config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]  # seconds between retries


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to the configured LLM and return the text response."""
    if settings.LLM_PROVIDER == "openai":
        return await _call_with_retry(_call_openai, system_prompt, user_prompt)
    elif settings.LLM_PROVIDER == "gemini":
        return await _call_with_retry(_call_gemini, system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")


async def _call_with_retry(func, system_prompt: str, user_prompt: str) -> str:
    """Retry LLM calls on rate limit (429) errors with exponential backoff."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return await func(system_prompt, user_prompt)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(f"Rate limited (429). Retrying in {delay}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                last_error = e
                await asyncio.sleep(delay)
            else:
                raise
    raise last_error


async def _call_openai(system_prompt: str, user_prompt: str) -> str:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "system_instruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": [
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ],
                "generationConfig": {"temperature": 0.1},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


def extract_json_from_response(text: str) -> dict | list:
    """Extract JSON from LLM response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # drop opening ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)