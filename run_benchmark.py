from pathlib import Path
import time
import json
from collections import Counter
from statistics import mean
from datetime import datetime

from llm_utils import load_prompt, query_model_chat, query_model_generate, safe_filename, save_result

MODELS = [
    "llama3.1:8b",
    #"Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0",
    #"qwen3:14b",
    #"Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0",
    #"deepseek-r1:8b",
    #"deepseek-r1:14b"
]

ATTEMPTS = 20
RESUME = True
LOG_FILE = Path("logs/benchmark.log")


def log_message(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{timestamp} | {message}"
    print(full_message)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(full_message + "\n")


def run_single_attempt(endpoint: str, model: str, system_prompt: str, user_prompt: str, attempt: int):
    if endpoint == "chat":
        return query_model_chat(model, system_prompt, user_prompt, attempt=attempt)
    elif endpoint == "generate":
        return query_model_generate(model, system_prompt, user_prompt, attempt=attempt)
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")


def normalize_error(error: str | None) -> str:
    if not error:
        return "none"

    error = error.strip()

    known_prefixes = [
        "Request timeout",
        "Request error",
        "Invalid JSON from model",
        "Unexpected error",
        "Model returned empty response",
        "Runner exception"
    ]

    for prefix in known_prefixes:
        if error.startswith(prefix):
            return prefix

    return error


def build_runtime_summary(endpoint_dir: Path) -> dict:
    attempt_files = sorted(endpoint_dir.glob("attempt_*.json"))

    if not attempt_files:
        return {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "success_rate": 0.0,
            "average_time_seconds": 0.0,
            "min_time_seconds": None,
            "max_time_seconds": None,
            "empty_raw_output_count": 0,
            "valid_json_count": 0,
            "invalid_json_count": 0,
            "error_types": {},
            "postprocessing_steps_counts_all": {},
            "postprocessing_steps_counts_success_only": {}
        }

    times = []
    success_count = 0
    failure_count = 0
    empty_raw_output_count = 0
    valid_json_count = 0
    invalid_json_count = 0
    error_counter = Counter()
    postprocessing_counter_all = Counter()
    postprocessing_counter_success_only = Counter()

    for file_path in attempt_files:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        time_value = data.get("time")
        if isinstance(time_value, (int, float)):
            times.append(time_value)

        success = bool(data.get("success"))
        if success:
            success_count += 1
            valid_json_count += 1
        else:
            failure_count += 1
            invalid_json_count += 1

        raw_output = data.get("raw_output")
        if raw_output is None or (isinstance(raw_output, str) and not raw_output.strip()):
            empty_raw_output_count += 1

        error_counter[normalize_error(data.get("error"))] += 1

        steps = data.get("postprocessing_steps", []) or []

        for step in steps:
            postprocessing_counter_all[step] += 1

        if success:
            for step in steps:
                postprocessing_counter_success_only[step] += 1

    return {
        "total_attempts": len(attempt_files),
        "successful_attempts": success_count,
        "failed_attempts": failure_count,
        "success_rate": round(success_count / len(attempt_files), 4),
        "average_time_seconds": round(mean(times), 4) if times else 0.0,
        "min_time_seconds": min(times) if times else None,
        "max_time_seconds": max(times) if times else None,
        "empty_raw_output_count": empty_raw_output_count,
        "valid_json_count": valid_json_count,
        "invalid_json_count": invalid_json_count,
        "error_types": dict(error_counter),
        "postprocessing_steps_counts_all": dict(postprocessing_counter_all),
        "postprocessing_steps_counts_success_only": dict(postprocessing_counter_success_only)
    }


def save_runtime_summary(endpoint_dir: Path) -> None:
    summary = build_runtime_summary(endpoint_dir)
    save_result(summary, str(endpoint_dir / "summary_runtime.json"))


def run_test_cerf():
    log_message("=== BENCHMARK STARTED ===")

    system_prompt = load_prompt("prompts/test/system_test.txt")
    num_sections = 7

    for model in MODELS:
        safe_model = safe_filename(model)
        log_message(f"MODEL STARTED | {model}")

        for section_num in range(1, num_sections + 1):
            user_prompt_path = f"prompts/test/user_test_section_{section_num}.txt"
            user_prompt = load_prompt(user_prompt_path)

            for endpoint in ["chat", "generate"]:
                endpoint_dir = Path(f"results/test_section_{section_num}/{safe_model}/{endpoint}")
                endpoint_dir.mkdir(parents=True, exist_ok=True)

                log_message(
                    f"SECTION STARTED | model={model} | section={section_num} | endpoint={endpoint}"
                )

                for attempt in range(1, ATTEMPTS + 1):
                    output_path = endpoint_dir / f"attempt_{attempt:03d}.json"

                    if RESUME and output_path.exists():
                        log_message(f"SKIP existing file | {output_path}")
                        continue

                    log_message(
                        f"RUN | model={model} | section={section_num} | endpoint={endpoint} | attempt={attempt}"
                    )

                    try:
                        result = run_single_attempt(
                            endpoint=endpoint,
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            attempt=attempt
                        )

                        result["mode_id"] = f"test_section_{section_num}"
                        result["endpoint"] = endpoint
                        result["system_prompt_path"] = "prompts/test/system_test.txt"
                        result["user_prompt_path"] = user_prompt_path
                        result["timestamp"] = datetime.now().isoformat(timespec="seconds")

                        save_result(result, str(output_path))

                    except Exception as e:
                        error_result = {
                            "attempt": attempt,
                            "model": model,
                            "time": None,
                            "success": False,
                            "response": None,
                            "raw_output": None,
                            "postprocessing_steps": [],
                            "error": f"Runner exception: {e}",
                            "mode_id": f"test_section_{section_num}",
                            "endpoint": endpoint,
                            "system_prompt_path": "prompts/test/system_test.txt",
                            "user_prompt_path": user_prompt_path,
                            "timestamp": datetime.now().isoformat(timespec="seconds")
                        }
                        save_result(error_result, str(output_path))

                    time.sleep(0.2)

                save_runtime_summary(endpoint_dir)
                log_message(
                    f"SUMMARY SAVED | model={model} | section={section_num} | endpoint={endpoint}"
                )

    log_message("=== BENCHMARK FINISHED ===")

def run_grammar():
    log_message("=== BENCHMARK STARTED ===")

    grammar_modes = [
        {
            "mode_id": "grammar_conditionals_1_2_3",
            "system_prompt_path": "prompts/grammar/system_grammar.txt",
            "user_prompt_path": "prompts/grammar/user_grammar_conditionals_1_2_3.txt",
        },
        {
            "mode_id": "grammar_gerund_vs_infinitive",
            "system_prompt_path": "prompts/grammar/system_grammar.txt",
            "user_prompt_path": "prompts/grammar/user_grammar_gerund_vs_infinitive.txt",
        },
        {
            "mode_id": "grammar_past_simple_vs_present_perfect",
            "system_prompt_path": "prompts/grammar/system_grammar.txt",
            "user_prompt_path": "prompts/grammar/user_grammar_past_vs_present_perfect.txt",
        },
        {
            "mode_id": "grammar_simple_vs_continuous",
            "system_prompt_path": "prompts/grammar/system_grammar.txt",
            "user_prompt_path": "prompts/grammar/user_grammar_simple_vs_continuous.txt",
        },
    ]

    for model in MODELS:
        safe_model = safe_filename(model)
        log_message(f"MODEL STARTED | {model}")

        for mode in grammar_modes:
            mode_id = mode["mode_id"]
            system_prompt_path = mode["system_prompt_path"]
            user_prompt_path = mode["user_prompt_path"]

            system_prompt = load_prompt(system_prompt_path)
            user_prompt = load_prompt(user_prompt_path)

            for endpoint in ["chat", "generate"]:
                endpoint_dir = Path(f"results/{mode_id}/{safe_model}/{endpoint}")
                endpoint_dir.mkdir(parents=True, exist_ok=True)

                log_message(
                    f"MODE STARTED | model={model} | mode={mode_id} | endpoint={endpoint}"
                )

                for attempt in range(1, ATTEMPTS + 1):
                    output_path = endpoint_dir / f"attempt_{attempt:03d}.json"

                    if RESUME and output_path.exists():
                        log_message(f"SKIP existing file | {output_path}")
                        continue

                    log_message(
                        f"RUN | model={model} | mode={mode_id} | endpoint={endpoint} | attempt={attempt}"
                    )

                    try:
                        result = run_single_attempt(
                            endpoint=endpoint,
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            attempt=attempt
                        )

                        result["mode_id"] = mode_id
                        result["endpoint"] = endpoint
                        result["system_prompt_path"] = system_prompt_path
                        result["user_prompt_path"] = user_prompt_path
                        result["timestamp"] = datetime.now().isoformat(timespec="seconds")

                        save_result(result, str(output_path))

                    except Exception as e:
                        error_result = {
                            "attempt": attempt,
                            "model": model,
                            "time": None,
                            "success": False,
                            "response": None,
                            "raw_output": None,
                            "postprocessing_steps": [],
                            "error": f"Runner exception: {e}",
                            "mode_id": mode_id,
                            "endpoint": endpoint,
                            "system_prompt_path": system_prompt_path,
                            "user_prompt_path": user_prompt_path,
                            "timestamp": datetime.now().isoformat(timespec="seconds")
                        }
                        save_result(error_result, str(output_path))

                    time.sleep(0.2)

                save_runtime_summary(endpoint_dir)
                log_message(
                    f"SUMMARY SAVED | model={model} | mode={mode_id} | endpoint={endpoint}"
                )

    log_message("=== BENCHMARK FINISHED ===")


def run_writing():
    log_message("=== BENCHMARK STARTED ===")

    mode_id = "writing"
    system_prompt_path = "prompts/writing/system_writing.txt"
    user_prompt_path = "prompts/writing/user_writing.txt"

    system_prompt = load_prompt(system_prompt_path)
    user_prompt = load_prompt(user_prompt_path)

    for model in MODELS:
        safe_model = safe_filename(model)
        log_message(f"MODEL STARTED | {model}")

        for endpoint in ["chat", "generate"]:
            endpoint_dir = Path(f"results/{mode_id}/{safe_model}/{endpoint}")
            endpoint_dir.mkdir(parents=True, exist_ok=True)

            log_message(
                f"MODE STARTED | model={model} | mode={mode_id} | endpoint={endpoint}"
            )

            for attempt in range(1, ATTEMPTS + 1):
                output_path = endpoint_dir / f"attempt_{attempt:03d}.json"

                if RESUME and output_path.exists():
                    log_message(f"SKIP existing file | {output_path}")
                    continue

                log_message(
                    f"RUN | model={model} | mode={mode_id} | endpoint={endpoint} | attempt={attempt}"
                )

                try:
                    result = run_single_attempt(
                        endpoint=endpoint,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        attempt=attempt
                    )

                    result["mode_id"] = mode_id
                    result["endpoint"] = endpoint
                    result["system_prompt_path"] = system_prompt_path
                    result["user_prompt_path"] = user_prompt_path
                    result["timestamp"] = datetime.now().isoformat(timespec="seconds")

                    save_result(result, str(output_path))

                except Exception as e:
                    error_result = {
                        "attempt": attempt,
                        "model": model,
                        "time": None,
                        "success": False,
                        "response": None,
                        "raw_output": None,
                        "postprocessing_steps": [],
                        "error": f"Runner exception: {e}",
                        "mode_id": mode_id,
                        "endpoint": endpoint,
                        "system_prompt_path": system_prompt_path,
                        "user_prompt_path": user_prompt_path,
                        "timestamp": datetime.now().isoformat(timespec="seconds")
                    }
                    save_result(error_result, str(output_path))

                time.sleep(0.2)

            save_runtime_summary(endpoint_dir)
            log_message(
                f"SUMMARY SAVED | model={model} | mode={mode_id} | endpoint={endpoint}"
            )

    log_message("=== BENCHMARK FINISHED ===")

def main():
    run_grammar()
    run_test_cerf()
    

if __name__ == "__main__":
    main()