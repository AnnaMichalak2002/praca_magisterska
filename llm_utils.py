import json
import re
import time
from pathlib import Path
from typing import Any
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

DEFAULT_TIMEOUT = 120

TEST_CEFR_OPTIONS = {
    "num_ctx": 2048,
    "num_predict": 2000,
    "temperature": 0.4,
    "top_p": 0.9,
}

WRITING_OPTIONS = {
    "num_ctx": 2048,
    "num_predict": 1200,
    "temperature": 0.0,
    "top_p": 0.9,
}

GRAMMAR_OPTIONS = {
    "num_ctx": 2048,
    "num_predict": 1500,
    "temperature": 0.3,
    "top_p": 0.9,
}

VOCABULARY_OPTIONS = {
    "num_ctx": 2048,
    "num_predict": 800,
    "temperature": 0.5,
    "top_p": 0.9,
}

DEFAULT_OPTIONS = TEST_CEFR_OPTIONS.copy()

DEFAULT_THINK = False

def get_options_for_mode(mode_id: str | None) -> dict:
    if not mode_id:
        return DEFAULT_OPTIONS.copy()

    if mode_id.startswith("test_section_"):
        return TEST_CEFR_OPTIONS.copy()

    if mode_id == "writing":
        return WRITING_OPTIONS.copy()

    if mode_id.startswith("grammar_"):
        return GRAMMAR_OPTIONS.copy()

    if mode_id.startswith("vocabulary_"):
        return VOCABULARY_OPTIONS.copy()

    return DEFAULT_OPTIONS.copy()

def safe_filename(name: str) -> str:
    """Replaces characters that are invalid in file names."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)

def load_prompt(path: str) -> str:
    """Loads prompt text from a UTF-8 encoded file."""
    return Path(path).read_text(encoding="utf-8")

def save_result(data: dict, filename: str) -> None:
    """Saves result data as formatted JSON, creating parent directories if needed."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_first_json_object(text: str) -> str:
    """
    Extracts the first complete JSON object from text,
    handling nesting, strings, and escaped characters.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    raise ValueError("No complete JSON object found")

def clean_json_response(raw_output: str) -> tuple[str, list[str]]:
    """
    Cleans raw model output by removing Markdown wrappers,
    extracting JSON, and recording cleanup steps.
    """
    steps: list[str] = []
    cleaned = raw_output.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
        steps.append("removed_markdown_code_fences")
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
        steps.append("removed_markdown_code_fences")

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
        if "removed_markdown_code_fences" not in steps:
            steps.append("removed_markdown_code_fences")

    extracted = extract_first_json_object(cleaned)

    start_idx = cleaned.find(extracted)
    end_idx = start_idx + len(extracted)

    if start_idx > 0 and cleaned[:start_idx].strip():
        steps.append("trimmed_leading_text_before_json")

    if end_idx < len(cleaned) and cleaned[end_idx:].strip():
        steps.append("trimmed_trailing_text_after_json")

    if not steps:
        steps.append("json_parsed_without_cleanup")

    return extracted, steps

def _extract_ollama_metrics(data: dict) -> dict:
    """
    Extracts performance and generation metadata returned by Ollama.
    Values are included as-is when present.
    """
    prompt_eval_count = data.get("prompt_eval_count")
    prompt_eval_duration = data.get("prompt_eval_duration")
    eval_count = data.get("eval_count")
    eval_duration = data.get("eval_duration")

    tokens_per_second = None
    if isinstance(eval_count, int) and isinstance(eval_duration, int) and eval_duration > 0:
        tokens_per_second = round(eval_count / (eval_duration / 1_000_000_000), 3)

    return {
        "created_at": data.get("created_at"),
        "done": data.get("done"),
        "done_reason": data.get("done_reason"),
        "total_duration_ns": data.get("total_duration"),
        "load_duration_ns": data.get("load_duration"),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_ns": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration,
        "tokens_per_second": tokens_per_second,
    }

def _build_base_result(
    *,
    attempt: int | None,
    model: str,
    elapsed_seconds: float,
    options: dict,
    timeout: int,
) -> dict:
    return {
        "attempt": attempt,
        "model": model,
        "endpoint": "chat",
        "time": round(elapsed_seconds, 3),
        "success": False,
        "response": None,
        "raw_output": None,
        "postprocessing_steps": [],
        "error": None,
        "request_options": options,
        "request_timeout_seconds": timeout,
        "http_status_code": None,
        "ollama_metrics": {},
    }

def _make_http_error_message(response: requests.Response) -> str:
    """
    Returns a readable HTTP error message with response body preview.
    """
    try:
        body_preview = response.text.strip()
    except Exception:
        body_preview = ""

    if len(body_preview) > 500:
        body_preview = body_preview[:500] + "... [truncated]"

    if body_preview:
        return f"HTTP {response.status_code}: {body_preview}"
    return f"HTTP {response.status_code}"

def query_model_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    attempt: int | None = None,
    mode_id: str | None = None,
    options: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT,
):
    """
    Sends messages to the Ollama /chat endpoint and attempts to parse the model output as JSON.
    """
    request_options = options.copy() if options is not None else get_options_for_mode(mode_id)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": DEFAULT_THINK,
        "options": request_options,
    }

    start = time.perf_counter()
    result = _build_base_result(
        attempt=attempt,
        model=model,
        elapsed_seconds=0.0,
        options=request_options,
        timeout=timeout,
    )

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start

        result["time"] = round(elapsed, 3)
        result["http_status_code"] = response.status_code

        if not response.ok:
            result["error"] = _make_http_error_message(response)
            return result

        data = response.json()
        result["ollama_metrics"] = _extract_ollama_metrics(data)

        raw_output = data.get("message", {}).get("content", "")
        result["raw_output"] = raw_output

        if not raw_output or not raw_output.strip():
            result["error"] = "Model returned empty response"
            return result

        cleaned_output, postprocessing_steps = clean_json_response(raw_output)
        parsed = json.loads(cleaned_output)

        result["success"] = True
        result["response"] = parsed
        result["postprocessing_steps"] = postprocessing_steps
        result["error"] = None
        return result

    except requests.exceptions.Timeout:
        result["time"] = round(time.perf_counter() - start, 3)
        result["error"] = "Request timeout"
        return result

    except requests.exceptions.RequestException as e:
        result["time"] = round(time.perf_counter() - start, 3)
        result["error"] = f"Request error: {e}"
        return result

    except json.JSONDecodeError as e:
        result["time"] = round(time.perf_counter() - start, 3)
        result["error"] = f"Invalid JSON from model: {e}"
        return result

    except Exception as e:
        result["time"] = round(time.perf_counter() - start, 3)
        result["error"] = f"Unexpected error: {e}"
        return result