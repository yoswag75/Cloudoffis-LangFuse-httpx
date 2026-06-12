import os
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes

load_dotenv()
lf = get_client()
DATASET_NAME = "gemma-httpx-ctransformers-conversations"


def ensure_dataset_exists():
    try:
        lf.get_dataset(DATASET_NAME)
    except Exception:
        lf.create_dataset(
            name=DATASET_NAME,
            description="Custom API + ctransformers — 4 endpoints",
        )


def get_prompt_and_config(prompt_name: str, label: str = "production"):
    prompt_obj = lf.get_prompt(prompt_name, label=label)
    config = prompt_obj.config or {}
    return {
        "prompt":      prompt_obj,
        "raw_text":    prompt_obj.prompt,
        "temperature": config.get("temperature", 0.7),
        "top_p":       config.get("top_p", 0.9),
        "top_k":       config.get("top_k", 40),
        "max_tokens":  config.get("max_tokens", 512),
    }


def save_turn_to_dataset(
    user_message: str,
    assistant_message: str,
    config: dict,
    session_id: str,
    endpoint: str,
):
    ensure_dataset_exists()
    lf.create_dataset_item(
        dataset_name=DATASET_NAME,
        input={
            "role":       "user",
            "content":    user_message,
            "endpoint":   endpoint,
            "session_id": session_id,
        },
        expected_output={
            "role":    "assistant",
            "content": assistant_message,
        },
        metadata={
            "api_mode":    "httpx → custom API",
            "backend":     "ctransformers",
            "endpoint":    endpoint,
            "model":       "gemma-4-e2b",
            "temperature": config["temperature"],
            "top_p":       config["top_p"],
            "top_k":       config["top_k"],
            "max_tokens":  config["max_tokens"],
            "session_id":  session_id,
        },
    )


def log_generation(
    session_id: str,
    messages: list,
    response: str,
    config: dict,
    latency: float,
    request_payload: dict,
    raw_response: dict,
    endpoint: str,
):
    prompt_tokens     = raw_response.get("prompt_tokens", 0)
    completion_tokens = raw_response.get("completion_tokens", 0)
    generation_time   = raw_response.get("generation_time_sec", 0)
    tokens_per_sec    = round(
        completion_tokens / generation_time, 2
    ) if generation_time and completion_tokens else None

    with lf.start_as_current_observation(
        as_type="span",
        name=f"chatbot-turn-{endpoint.split('/')[-1]}",
    ):
        with propagate_attributes(
            user_id="streamlit-user",
            session_id=session_id,
            tags=["httpx", "custom-api", "ctransformers", endpoint],
            metadata={
                "api_mode":       "httpx → custom API",
                "backend":        "ctransformers",
                "endpoint":       endpoint,
                "model":          "gemma-4-e2b",
                "temperature":    config["temperature"],
                "top_p":          config["top_p"],
                "top_k":          config["top_k"],
                "max_tokens":     config["max_tokens"],
                "latency_sec":    latency,
                "tokens_per_sec": tokens_per_sec,
            },
        ):
            with lf.start_as_current_observation(
                as_type="span",
                name=f"http-request-{endpoint.split('/')[-1]}",
                input={
                    "method":  "POST",
                    "url":     f"{os.getenv('CUSTOM_API_URL')}{endpoint}",
                    "payload": request_payload,
                },
            ) as req_span:
                req_span.update(
                    output={"status_code": 200, "tokens_per_sec": tokens_per_sec},
                    metadata={"latency_sec": latency},
                )

            with lf.start_as_current_observation(
                as_type="generation",
                name=f"llm-call-{endpoint.split('/')[-1]}",
                model="gemma-4-e2b",
                input=messages,
                model_parameters={
                    "temperature":    config["temperature"],
                    "top_p":          config["top_p"],
                    "top_k":          config["top_k"],
                    "max_new_tokens": config["max_tokens"],
                    "backend":        "ctransformers",
                    "endpoint":       endpoint,
                },
            ) as gen:
                gen.update(
                    output=response,
                    usage={
                        "input":  prompt_tokens,
                        "output": completion_tokens,
                        "total":  prompt_tokens + completion_tokens,
                    },
                    metadata={
                        "latency_sec":        latency,
                        "generation_time_sec": generation_time,
                        "tokens_per_sec":     tokens_per_sec,
                        "prompt_tokens":      prompt_tokens,
                        "completion_tokens":  completion_tokens,
                    },
                )

    lf.flush()