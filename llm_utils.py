import requests
import json
import time
from pathlib import Path
import sys

OLLAMA_URL_prompt = "http://localhost:11434/api/generate"
OLLAMA_URL_message = "http://localhost:11434/api/chat"

def safe_filename(name: str) -> str:
    """Replaces characters that are invalid in file names."""
    return name.replace(":", "_").replace("/", "_")
    #re.sub(r'[<>:"/\\|?*]', "_", name)

def load_prompt(path: str) -> str:
    """Loads prompt text from a UTF-8 encoded file."""
    return Path(path).read_text(encoding="utf-8")

def save_result(data, filename) -> None:
    """Saves result data as formatted JSON, creating parent directories if needed."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def build_prompt(system: str, user: str) -> str:
    """Builds a prompt in SYSTEM/USER text format."""
    parts = [
        "SYSTEM:",
        system.strip(),
        "",
        "USER:",
        user.strip(),
    ]
    return "\n".join(parts)

def extract_first_json_object(text: str) -> str:
    """Extracts the first complete JSON object from text, handling nesting, strings, and escaped characters."""
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
    """Cleans raw model output by removing Markdown wrappers, extracting JSON, and recording cleanup steps."""
    steps = []
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

def query_model_generate(model: str, system_prompt: str, user_prompt: str, attempt: int | None = None):
    """
    Sends a prompt to the Ollama /generate endpoint and attempts to parse the model output as JSON.

    The function builds a combined prompt from the system and user messages, sends it to the model,
    measures response time, cleans the returned content from Markdown code fences if needed,
    and validates whether the output is a valid JSON object.

    Returns:
        dict: A dictionary containing:
            - attempt: test iteration number,
            - model: model name,
            - time: response time in seconds,
            - success: whether the request and JSON parsing succeeded,
            - response: parsed JSON response or None,
            - raw_output: raw model output,
            - error: error message or None.
    """

    prompt = build_prompt(system_prompt, user_prompt)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    start = time.perf_counter()

    try:
        response = requests.post(OLLAMA_URL_prompt, json=payload, timeout=600)
        response.raise_for_status()

        data = response.json()
        raw_output = data.get("response", "")

        if not raw_output or not raw_output.strip():
            return {
                "attempt": attempt,
                "model": model,
                "time": round(time.perf_counter() - start, 3),
                "success": False,
                "response": None,
                "raw_output": raw_output,
                "postprocessing_steps": [],
                "error": "Model returned empty response"
            }

        cleaned_output, postprocessing_steps = clean_json_response(raw_output)
        parsed = json.loads(cleaned_output)

        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": True,
            "response": parsed,
            "raw_output": raw_output,
            "postprocessing_steps": postprocessing_steps,
            "error": None
        }

    except requests.exceptions.Timeout:
        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": False,
            "response": None,
            "raw_output": None,
            "postprocessing_steps": postprocessing_steps if 'postprocessing_steps' in locals() else [],
            "error": "Request timeout"
        }

    except requests.exceptions.RequestException as e:
        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": False,
            "response": None,
            "raw_output": None,
            "postprocessing_steps": postprocessing_steps if 'postprocessing_steps' in locals() else [],
            "error": f"Request error: {e}"
        }

    except json.JSONDecodeError as e:
        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": False,
            "response": None,
            "raw_output": raw_output if 'raw_output' in locals() else None,
            "postprocessing_steps": postprocessing_steps if 'postprocessing_steps' in locals() else [],
            "error": f"Invalid JSON from model: {e}"
        }

    except Exception as e:
        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": False,
            "response": None,
            "raw_output": raw_output if 'raw_output' in locals() else None,
            "postprocessing_steps": postprocessing_steps if 'postprocessing_steps' in locals() else [],
            "error": f"Unexpected error: {e}"
        }
    
def query_model_chat(model: str, system_prompt: str, user_prompt: str, attempt: int | None = None):
    """
    Sends messages to the Ollama /chat endpoint and attempts to parse the model output as JSON.

    The function sends the system and user prompts in chat format, measures response time,
    extracts the assistant's message content, removes Markdown code fences if present,
    and checks whether the returned content can be parsed as valid JSON.

    Returns:
        dict: A dictionary containing:
            - attempt: test iteration number,
            - model: model name,
            - time: response time in seconds,
            - success: whether the request and JSON parsing succeeded,
            - response: parsed JSON response or None,
            - raw_output: raw model output,
            - error: error message or None.
    """

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False
    }

    start = time.perf_counter()

    try:
        response = requests.post(OLLAMA_URL_message, json=payload, timeout=600)
        response.raise_for_status()

        data = response.json()
        raw_output = data.get("message", {}).get("content", "")

        if not raw_output or not raw_output.strip():
            return {
                "attempt": attempt,
                "model": model,
                "time": round(time.perf_counter() - start, 3),
                "success": False,
                "response": None,
                "raw_output": raw_output,
                "postprocessing_steps": [],
                "error": "Model returned empty response"
            }

        cleaned_output, postprocessing_steps = clean_json_response(raw_output)
        parsed = json.loads(cleaned_output)

        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": True,
            "response": parsed,
            "raw_output": raw_output,
            "postprocessing_steps": postprocessing_steps,
            "error": None
        }

    except requests.exceptions.Timeout:
        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": False,
            "response": None,
            "raw_output": None,
            "postprocessing_steps": postprocessing_steps if 'postprocessing_steps' in locals() else [],
            "error": "Request timeout"
        }

    except requests.exceptions.RequestException as e:
        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": False,
            "response": None,
            "raw_output": None,
            "postprocessing_steps": postprocessing_steps if 'postprocessing_steps' in locals() else [],
            "error": f"Request error: {e}"
        }

    except json.JSONDecodeError as e:
        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": False,
            "response": None,
            "raw_output": raw_output if 'raw_output' in locals() else None,
            "postprocessing_steps": postprocessing_steps if 'postprocessing_steps' in locals() else [],
            "error": f"Invalid JSON from model: {e}"
        }

    except Exception as e:
        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": False,
            "response": None,
            "raw_output": raw_output if 'raw_output' in locals() else None,
            "postprocessing_steps": postprocessing_steps if 'postprocessing_steps' in locals() else [],
            "error": f"Unexpected error: {e}"
        }