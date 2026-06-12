import os
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = None


def get_base_url():
    global BASE_URL
    if BASE_URL is None:
        BASE_URL = os.getenv("CUSTOM_API_URL", "http://api:8000")
    return BASE_URL


def call_chat(
    history: list,
    config: dict,
    system_prompt: str,
    user_message: str,
) -> tuple[str, dict, dict]:
    """Endpoint: POST /api/v1/chat — multi-turn conversation."""
    url = get_base_url() + "/api/v1/chat"

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "messages":       messages,
        "temperature":    config["temperature"],
        "top_p":          config["top_p"],
        "top_k":          config["top_k"],
        "max_new_tokens": config["max_tokens"],
    }

    return _post(url, payload, "response")


def call_complete(
    prompt: str,
    config: dict,
) -> tuple[str, dict, dict]:
    """Endpoint: POST /api/v1/complete — raw single-turn completion."""
    url = get_base_url() + "/api/v1/complete"

    payload = {
        "prompt":         prompt,
        "temperature":    config["temperature"],
        "top_p":          config["top_p"],
        "top_k":          config["top_k"],
        "max_new_tokens": config["max_tokens"],
    }

    return _post(url, payload, "response")


def call_summarize(
    text: str,
    style: str,
    config: dict,
) -> tuple[str, dict, dict]:
    """Endpoint: POST /api/v1/summarize — dedicated summarization."""
    url = get_base_url() + "/api/v1/summarize"

    payload = {
        "text":           text,
        "style":          style,
        "temperature":    config["temperature"],
        "top_p":          config["top_p"],
        "top_k":          config["top_k"],
        "max_new_tokens": config["max_tokens"],
    }

    return _post(url, payload, "summary")


def call_instruct(
    instruction: str,
    context: str,
    config: dict,
) -> tuple[str, dict, dict]:
    """Endpoint: POST /api/v1/instruct — instruction + context."""
    url = get_base_url() + "/api/v1/instruct"

    payload = {
        "instruction":    instruction,
        "context":        context,
        "temperature":    config["temperature"],
        "top_p":          config["top_p"],
        "top_k":          config["top_k"],
        "max_new_tokens": config["max_tokens"],
    }

    return _post(url, payload, "response")


def _post(url: str, payload: dict, response_key: str) -> tuple[str, dict, dict]:
    """Shared HTTP POST logic."""
    with httpx.Client(timeout=300) as client:
        r = client.post(url, json=payload)

        if r.status_code != 200:
            raise ValueError(f"API error {r.status_code}: {r.text}")

        raw = r.json()
        content = raw.get(response_key, "").strip()

        if not content:
            raise ValueError(f"Empty response. Raw: {raw}")

    return content, payload, raw