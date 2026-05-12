from pathlib import Path
import re
import time
import json
import csv
import subprocess
import threading
from collections import Counter
from statistics import mean
from datetime import datetime

from llm_utils import load_prompt, query_model_chat, query_model_generate, safe_filename, save_result

MODELS = [
    "ministral-3:3b",
    "qwen3.5:4b",
    "llama3.1:8b",
    "Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0",
    "deepseek-r1:8b",
]

ATTEMPTS = 2
RESUME = True
LOG_FILE = Path("logs/benchmark.log")

GPU_SAMPLING_ENABLED = True
GPU_SAMPLE_INTERVAL_SECONDS = 1.0

SLEEP_BETWEEN_ATTEMPTS_SECONDS = 5.0
SLEEP_BETWEEN_ENDPOINTS_SECONDS = 8.0
SLEEP_BETWEEN_MODELS_SECONDS = 12.0

SAVE_GPU_SAMPLES_MODE = "all"

def should_save_gpu_samples(attempt: int) -> bool:
    if SAVE_GPU_SAMPLES_MODE == "all":
        return True
    if SAVE_GPU_SAMPLES_MODE == "none":
        return False
    if SAVE_GPU_SAMPLES_MODE == "first_attempt_only":
        return attempt == 1
    return False

def sleep_between_attempts():
    time.sleep(SLEEP_BETWEEN_ATTEMPTS_SECONDS)


def sleep_between_endpoints():
    time.sleep(SLEEP_BETWEEN_ENDPOINTS_SECONDS)


def sleep_between_models():
    time.sleep(SLEEP_BETWEEN_MODELS_SECONDS)

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
        "Runner exception",
        "HTTP"
    ]

    for prefix in known_prefixes:
        if error.startswith(prefix):
            return prefix

    return error


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _split_csv_line(line: str) -> list[str]:
    reader = csv.reader([line])
    return next(reader)


def get_ollama_ps_snapshot(model_name: str) -> dict:
    """
    Returns runtime information for a loaded model from `ollama ps`.

    Expected row examples:
    ministral-3:3b    f04aa1c738f6    4.7 GB    100% GPU          2048    4 minutes from now
    qwen3.5:4b        2a654d98e6fb    6.2 GB    30%/70% CPU/GPU   2048    4 minutes from now
    """

    try:
        completed = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )

        if completed.returncode != 0:
            return {
                "available": False,
                "error": f"ollama ps failed: {completed.stderr.strip() or completed.stdout.strip()}",
            }

        lines = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) < 2:
            return {
                "available": False,
                "error": "ollama ps returned no running models",
            }

        target_line = None
        for line in lines[1:]:
            if line.startswith(model_name):
                target_line = line
                break

        if target_line is None:
            return {
                "available": False,
                "error": f"model not found in ollama ps: {model_name}",
            }

        columns = re.split(r"\s{2,}", target_line.strip())
        
        if len(columns) < 6:
            return {
                "available": False,
                "raw_line": target_line,
                "parsed_columns": columns,
                "error": "unexpected ollama ps format",
            }

        name = columns[0]
        model_id = columns[1]
        loaded_size = columns[2]
        processor = columns[3]
        context = columns[4]
        until = columns[5]

        return {
            "available": True,
            "name": name,
            "id": model_id,
            "loaded_size": loaded_size,
            "processor": processor,
            "context": context,
            "until": until,
            "raw_line": target_line,
            "parsed_columns": columns,
        }

    except Exception as e:
        return {
            "available": False,
            "error": f"ollama ps exception: {e}",
        }


def sample_nvidia_smi_once() -> dict:
    """
    Takes a single GPU metrics snapshot using nvidia-smi.
    """
    query = (
        "timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    )

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )

        if completed.returncode != 0:
            return {
                "available": False,
                "error": f"nvidia-smi failed: {completed.stderr.strip() or completed.stdout.strip()}",
            }

        line = completed.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in _split_csv_line(line)]

        if len(parts) < 7:
            return {
                "available": False,
                "raw_line": line,
                "error": "unexpected nvidia-smi output format",
            }

        return {
            "available": True,
            "timestamp": parts[0],
            "gpu_name": parts[1],
            "gpu_utilization_percent": _safe_float(parts[2]),
            "memory_used_mib": _safe_float(parts[3]),
            "memory_total_mib": _safe_float(parts[4]),
            "temperature_c": _safe_float(parts[5]),
            "power_draw_w": _safe_float(parts[6]),
            "raw_line": line,
        }

    except Exception as e:
        return {
            "available": False,
            "error": f"nvidia-smi exception: {e}",
        }


def collect_gpu_samples_during_run(stop_event: threading.Event, samples: list[dict]) -> None:
    """
    Collects GPU samples in the background until stop_event is set.
    """
    while not stop_event.is_set():
        sample = sample_nvidia_smi_once()
        sample["collected_at_local"] = datetime.now().isoformat(timespec="seconds")
        samples.append(sample)

        if stop_event.wait(GPU_SAMPLE_INTERVAL_SECONDS):
            break


def build_gpu_summary(samples: list[dict]) -> dict:
    valid_samples = [s for s in samples if s.get("available")]

    if not valid_samples:
        return {
            "sample_count": len(samples),
            "valid_sample_count": 0,
            "avg_gpu_utilization_percent": None,
            "max_gpu_utilization_percent": None,
            "avg_memory_used_mib": None,
            "max_memory_used_mib": None,
            "avg_temperature_c": None,
            "max_temperature_c": None,
            "avg_power_draw_w": None,
            "max_power_draw_w": None,
            "estimated_gpu_energy_wh": None,
        }

    gpu_utils = [s["gpu_utilization_percent"] for s in valid_samples if s.get("gpu_utilization_percent") is not None]
    mem_used = [s["memory_used_mib"] for s in valid_samples if s.get("memory_used_mib") is not None]
    temps = [s["temperature_c"] for s in valid_samples if s.get("temperature_c") is not None]
    powers = [s["power_draw_w"] for s in valid_samples if s.get("power_draw_w") is not None]

    estimated_gpu_energy_wh = None
    if powers:
        avg_power = mean(powers)
        total_seconds = len(powers) * GPU_SAMPLE_INTERVAL_SECONDS
        estimated_gpu_energy_wh = round(avg_power * (total_seconds / 3600), 6)

    return {
        "sample_count": len(samples),
        "valid_sample_count": len(valid_samples),
        "avg_gpu_utilization_percent": round(mean(gpu_utils), 3) if gpu_utils else None,
        "max_gpu_utilization_percent": max(gpu_utils) if gpu_utils else None,
        "avg_memory_used_mib": round(mean(mem_used), 3) if mem_used else None,
        "max_memory_used_mib": max(mem_used) if mem_used else None,
        "avg_temperature_c": round(mean(temps), 3) if temps else None,
        "max_temperature_c": max(temps) if temps else None,
        "avg_power_draw_w": round(mean(powers), 3) if powers else None,
        "max_power_draw_w": max(powers) if powers else None,
        "estimated_gpu_energy_wh": estimated_gpu_energy_wh,
    }


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
            "postprocessing_steps_counts_success_only": {},
            "avg_tokens_per_second_success_only": None,
            "avg_total_duration_ns_success_only": None,
            "avg_eval_count_success_only": None,
            "avg_prompt_eval_count_success_only": None,
            "avg_gpu_utilization_percent": None,
            "avg_gpu_memory_used_mib": None,
            "avg_gpu_power_draw_w": None,
            "avg_gpu_energy_wh": None,
            "processor_distribution": {},
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

    tokens_per_second_values = []
    total_duration_values = []
    eval_count_values = []
    prompt_eval_count_values = []

    avg_gpu_util_values = []
    avg_gpu_mem_values = []
    avg_gpu_power_values = []
    gpu_energy_values = []

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

        ollama_metrics = data.get("ollama_metrics", {}) or {}
        if success:
            if isinstance(ollama_metrics.get("tokens_per_second"), (int, float)):
                tokens_per_second_values.append(ollama_metrics["tokens_per_second"])
            if isinstance(ollama_metrics.get("total_duration_ns"), int):
                total_duration_values.append(ollama_metrics["total_duration_ns"])
            if isinstance(ollama_metrics.get("eval_count"), int):
                eval_count_values.append(ollama_metrics["eval_count"])
            if isinstance(ollama_metrics.get("prompt_eval_count"), int):
                prompt_eval_count_values.append(ollama_metrics["prompt_eval_count"])

        gpu_summary = data.get("gpu_summary", {}) or {}
        if isinstance(gpu_summary.get("avg_gpu_utilization_percent"), (int, float)):
            avg_gpu_util_values.append(gpu_summary["avg_gpu_utilization_percent"])
        if isinstance(gpu_summary.get("avg_memory_used_mib"), (int, float)):
            avg_gpu_mem_values.append(gpu_summary["avg_memory_used_mib"])
        if isinstance(gpu_summary.get("avg_power_draw_w"), (int, float)):
            avg_gpu_power_values.append(gpu_summary["avg_power_draw_w"])
        if isinstance(gpu_summary.get("estimated_gpu_energy_wh"), (int, float)):
            gpu_energy_values.append(gpu_summary["estimated_gpu_energy_wh"])

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
        "postprocessing_steps_counts_success_only": dict(postprocessing_counter_success_only),
        "avg_tokens_per_second_success_only": round(mean(tokens_per_second_values), 4) if tokens_per_second_values else None,
        "avg_total_duration_ns_success_only": round(mean(total_duration_values), 2) if total_duration_values else None,
        "avg_eval_count_success_only": round(mean(eval_count_values), 4) if eval_count_values else None,
        "avg_prompt_eval_count_success_only": round(mean(prompt_eval_count_values), 4) if prompt_eval_count_values else None,
        "avg_gpu_utilization_percent": round(mean(avg_gpu_util_values), 4) if avg_gpu_util_values else None,
        "avg_gpu_memory_used_mib": round(mean(avg_gpu_mem_values), 4) if avg_gpu_mem_values else None,
        "avg_gpu_power_draw_w": round(mean(avg_gpu_power_values), 4) if avg_gpu_power_values else None,
        "avg_gpu_energy_wh": round(mean(gpu_energy_values), 6) if gpu_energy_values else None,
    }


def save_runtime_summary(endpoint_dir: Path, model: str) -> None:
    summary = build_runtime_summary(endpoint_dir)

    try:
        summary["ollama_ps_snapshot"] = get_ollama_ps_snapshot(model)
    except Exception as e:
        summary["ollama_ps_snapshot"] = {
            "available": False,
            "error": f"summary snapshot exception: {e}",
        }

    save_result(summary, str(endpoint_dir / "summary_runtime.json"))


def execute_attempt_with_monitoring(
    *,
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    attempt: int,
) -> dict:
    gpu_samples = []
    stop_event = threading.Event()
    monitor_thread = None

    if GPU_SAMPLING_ENABLED:
        monitor_thread = threading.Thread(
            target=collect_gpu_samples_during_run,
            args=(stop_event, gpu_samples),
            daemon=True,
        )
        monitor_thread.start()

    try:
        result = run_single_attempt(
            endpoint=endpoint,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            attempt=attempt,
        )
    finally:
        if GPU_SAMPLING_ENABLED:
            stop_event.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=3)

    result["gpu_summary"] = build_gpu_summary(gpu_samples)

    if should_save_gpu_samples(attempt):
        result["gpu_samples"] = gpu_samples
    else:
        result["gpu_samples"] = []

    return result


def run_test_cerf():
    log_message("=== BENCHMARK STARTED ===")

    system_prompt = load_prompt("prompts/test/system_test.txt")
    num_sections = 7
    endpoints = ["chat", "generate"]

    for model_idx, model in enumerate(MODELS):
        safe_model = safe_filename(model)
        log_message(f"MODEL STARTED | {model}")
        model_had_executed_attempt = False

        for section_num in range(1, num_sections + 1):
            user_prompt_path = f"prompts/test/user_test_section_{section_num}.txt"
            user_prompt = load_prompt(user_prompt_path)

            for endpoint_idx, endpoint in enumerate(endpoints):
                endpoint_dir = Path(f"results/test_section_{section_num}/{safe_model}/{endpoint}")
                endpoint_dir.mkdir(parents=True, exist_ok=True)

                log_message(
                    f"SECTION STARTED | model={model} | section={section_num} | endpoint={endpoint}"
                )

                endpoint_had_executed_attempt = False

                for attempt in range(1, ATTEMPTS + 1):
                    output_path = endpoint_dir / f"attempt_{attempt:03d}.json"
                    attempt_executed = False

                    if RESUME and output_path.exists():
                        log_message(f"SKIP existing file | {output_path}")
                    else:
                        attempt_executed = True
                        endpoint_had_executed_attempt = True
                        model_had_executed_attempt = True
                        
                        log_message(
                            f"RUN | model={model} | section={section_num} | endpoint={endpoint} | attempt={attempt}"
                        )

                        try:
                            result = execute_attempt_with_monitoring(
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
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "gpu_samples": [],
                                "gpu_summary": build_gpu_summary([]),
                            }
                            save_result(error_result, str(output_path))

                    if attempt_executed and attempt < ATTEMPTS:
                        sleep_between_attempts()

                save_runtime_summary(endpoint_dir, model)
                log_message(
                    f"SUMMARY SAVED | model={model} | section={section_num} | endpoint={endpoint}"
                )

                if endpoint_idx < len(endpoints) - 1 and endpoint_had_executed_attempt:
                    sleep_between_endpoints()

        if model_idx < len(MODELS) - 1 and model_had_executed_attempt:
            sleep_between_models()

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

    endpoints = ["chat", "generate"]

    for model_idx, model in enumerate(MODELS):
        safe_model = safe_filename(model)
        log_message(f"MODEL STARTED | {model}")
        model_had_executed_attempt = False

        for mode in grammar_modes:
            mode_id = mode["mode_id"]
            system_prompt_path = mode["system_prompt_path"]
            user_prompt_path = mode["user_prompt_path"]

            system_prompt = load_prompt(system_prompt_path)
            user_prompt = load_prompt(user_prompt_path)

            for endpoint_idx, endpoint in enumerate(endpoints):
                endpoint_dir = Path(f"results/{mode_id}/{safe_model}/{endpoint}")
                endpoint_dir.mkdir(parents=True, exist_ok=True)

                log_message(
                    f"MODE STARTED | model={model} | mode={mode_id} | endpoint={endpoint}"
                )

                endpoint_had_executed_attempt = False

                for attempt in range(1, ATTEMPTS + 1):
                    output_path = endpoint_dir / f"attempt_{attempt:03d}.json"
                    attempt_executed = False

                    if RESUME and output_path.exists():
                        log_message(f"SKIP existing file | {output_path}")
                    else:
                        attempt_executed = True
                        endpoint_had_executed_attempt = True
                        model_had_executed_attempt = True

                        log_message(
                            f"RUN | model={model} | mode={mode_id} | endpoint={endpoint} | attempt={attempt}"
                        )

                        try:
                            result = execute_attempt_with_monitoring(
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
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "gpu_samples": [],
                                "gpu_summary": build_gpu_summary([]),
                            }
                            save_result(error_result, str(output_path))

                    if attempt_executed and attempt < ATTEMPTS:
                        sleep_between_attempts()

                save_runtime_summary(endpoint_dir, model)
                log_message(
                    f"SUMMARY SAVED | model={model} | mode={mode_id} | endpoint={endpoint}"
                )

                if endpoint_idx < len(endpoints) - 1 and endpoint_had_executed_attempt:
                    sleep_between_endpoints()

        if model_idx < len(MODELS) - 1 and model_had_executed_attempt:
            sleep_between_models()

    log_message("=== BENCHMARK FINISHED ===")


def run_writing():
    log_message("=== BENCHMARK STARTED ===")

    mode_id = "writing"
    system_prompt_path = "prompts/writing/system_writing.txt"
    user_prompt_path = "prompts/writing/user_writing.txt"

    system_prompt = load_prompt(system_prompt_path)
    user_prompt = load_prompt(user_prompt_path)
    endpoints = ["chat", "generate"]

    for model_idx, model in enumerate(MODELS):
        safe_model = safe_filename(model)
        log_message(f"MODEL STARTED | {model}")
        model_had_executed_attempt = False

        for endpoint_idx, endpoint in enumerate(endpoints):
            endpoint_dir = Path(f"results/{mode_id}/{safe_model}/{endpoint}")
            endpoint_dir.mkdir(parents=True, exist_ok=True)

            log_message(
                f"MODE STARTED | model={model} | mode={mode_id} | endpoint={endpoint}"
            )
            endpoint_had_executed_attempt = False

            for attempt in range(1, ATTEMPTS + 1):
                output_path = endpoint_dir / f"attempt_{attempt:03d}.json"
                attempt_executed = False

                if RESUME and output_path.exists():
                    log_message(f"SKIP existing file | {output_path}")
                else:
                    attempt_executed = True
                    endpoint_had_executed_attempt = True
                    model_had_executed_attempt = True
                    log_message(
                        f"RUN | model={model} | mode={mode_id} | endpoint={endpoint} | attempt={attempt}"
                    )

                    try:
                        result = execute_attempt_with_monitoring(
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
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "gpu_samples": [],
                            "gpu_summary": build_gpu_summary([]),
                        }
                        save_result(error_result, str(output_path))

                if attempt_executed and attempt < ATTEMPTS:
                    sleep_between_attempts()

            save_runtime_summary(endpoint_dir, model)
            log_message(
                f"SUMMARY SAVED | model={model} | mode={mode_id} | endpoint={endpoint}"
            )

            if endpoint_idx < len(endpoints) - 1 and endpoint_had_executed_attempt:
                sleep_between_endpoints()

        if model_idx < len(MODELS) - 1 and model_had_executed_attempt:
            sleep_between_models()

    log_message("=== BENCHMARK FINISHED ===")


def run_vocabulary():
    log_message("=== BENCHMARK STARTED ===")

    vocabulary_modes = [
        {
            "mode_id": "vocabulary_definition",
            "system_prompt_path": "prompts/vocabulary/system_vocabulary.txt",
            "user_prompt_path": "prompts/vocabulary/user_vocabulary_definition.txt",
        },
        {
            "mode_id": "vocabulary_match",
            "system_prompt_path": "prompts/vocabulary/system_vocabulary.txt",
            "user_prompt_path": "prompts/vocabulary/user_vocabulary_match.txt",
        },
        {
            "mode_id": "vocabulary_synonym",
            "system_prompt_path": "prompts/vocabulary/system_vocabulary.txt",
            "user_prompt_path": "prompts/vocabulary/user_vocabulary_synonym.txt",
        },
        {
            "mode_id": "vocabulary_antonym",
            "system_prompt_path": "prompts/vocabulary/system_vocabulary.txt",
            "user_prompt_path": "prompts/vocabulary/user_vocabulary_antonym.txt",
        },
    ]

    endpoints = ["chat", "generate"]

    for model_idx, model in enumerate(MODELS):
        safe_model = safe_filename(model)
        log_message(f"MODEL STARTED | {model}")
        model_had_executed_attempt = False

        for mode in vocabulary_modes:
            mode_id = mode["mode_id"]
            system_prompt_path = mode["system_prompt_path"]
            user_prompt_path = mode["user_prompt_path"]

            system_prompt = load_prompt(system_prompt_path)
            user_prompt = load_prompt(user_prompt_path)

            for endpoint_idx, endpoint in enumerate(endpoints):
                endpoint_dir = Path(f"results/{mode_id}/{safe_model}/{endpoint}")
                endpoint_dir.mkdir(parents=True, exist_ok=True)

                log_message(
                    f"MODE STARTED | model={model} | mode={mode_id} | endpoint={endpoint}"
                )

                endpoint_had_executed_attempt = False
                
                for attempt in range(1, ATTEMPTS + 1):
                    output_path = endpoint_dir / f"attempt_{attempt:03d}.json"
                    attempt_executed = False

                    if RESUME and output_path.exists():
                        log_message(f"SKIP existing file | {output_path}")
                    else:
                        attempt_executed = True
                        endpoint_had_executed_attempt = True
                        model_had_executed_attempt = True
                        
                        log_message(
                            f"RUN | model={model} | mode={mode_id} | endpoint={endpoint} | attempt={attempt}"
                        )

                        try:
                            result = execute_attempt_with_monitoring(
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
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "gpu_samples": [],
                                "gpu_summary": build_gpu_summary([]),
                            }
                            save_result(error_result, str(output_path))

                    if attempt_executed and attempt < ATTEMPTS:
                        sleep_between_attempts()

                save_runtime_summary(endpoint_dir, model)
                log_message(
                    f"SUMMARY SAVED | model={model} | mode={mode_id} | endpoint={endpoint}"
                )

                if endpoint_idx < len(endpoints) - 1 and endpoint_had_executed_attempt:
                    sleep_between_endpoints()

        if model_idx < len(MODELS) - 1 and model_had_executed_attempt:
            sleep_between_models()

    log_message("=== BENCHMARK FINISHED ===")


def main():
    #run_vocabulary()
    #run_grammar()
    #run_test_cerf()
    run_writing()
    #run_grammar()
    run_vocabulary()


if __name__ == "__main__":
    main()