import requests
import json
import time
from pathlib import Path
import sys

OLLAMA_URL_prompt = "http://localhost:11434/api/generate"
OLLAMA_URL_message = "http://localhost:11434/api/chat"

MODELS = ["llama3.1:8b", "qwen3:14b", "Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0","deepseek-r1:8b", "deepseek-r1:14b" ]

def safe_filename(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")

def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def save_result(data, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def build_prompt(system: str, user: str) -> str:
    parts = [
        "SYSTEM:",
        system.strip(),
        "",
        "USER:",
        user.strip(),
    ]
    return "\n".join(parts)


def clean_json_response(raw_output: str) -> str:
    cleaned = raw_output.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned

def query_model_generate(model: str, system_prompt: str, user_prompt: str, attempt: int | None = None):
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
                "error": "Model returned empty response"
            }

        cleaned_output = clean_json_response(raw_output)
        parsed = json.loads(cleaned_output)

        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": True,
            "response": parsed,
            "raw_output": raw_output,
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
            "error": f"Unexpected error: {e}"
        }
    
def query_model_chat(model: str, system_prompt: str, user_prompt: str, attempt: int | None = None):

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
                "error": "Model returned empty response"
            }

        cleaned_output = clean_json_response(raw_output)
        parsed = json.loads(cleaned_output)

        return {
            "attempt": attempt,
            "model": model,
            "time": round(time.perf_counter() - start, 3),
            "success": True,
            "response": parsed,
            "raw_output": raw_output,
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
            "error": f"Unexpected error: {e}"
        }


if __name__ == "__main__":

    #model = "Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0"
    model = "llama3.1:8b"

    system_prompt = load_prompt("prompts/writing/system_writing.txt")
    user_prompt = load_prompt("prompts/writing/user_writing.txt")

    result = query_model_chat(model, system_prompt, user_prompt)

    print("Czas:", result["time"], "s")
    print(result["response"])

    safe_model = safe_filename(model)
    save_result(result, f"results/writing/chat/writing_{safe_model}.json")

    result2 = query_model_generate(model, system_prompt, user_prompt)

    print("Czas:", result2["time"], "s")
    print(result2["response"])

    safe_model = safe_filename(model)
    save_result(result2, f"results/writing/generate/writing_{safe_model}.json")

    sys.exit()