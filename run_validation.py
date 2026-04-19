from pathlib import Path
import json
from collections import Counter
from datetime import datetime

from llm_utils import save_result
from validators.specs import is_test_mode, is_grammar_mode, is_writing_mode
from validators.strict_validators import validate_test_strict, validate_grammar_strict, validate_writing_strict
from validators.tolerant_validators import validate_test_tolerant, validate_grammar_tolerant, validate_writing_tolerant


LOG_FILE = Path("logs/run_validation.log")


def log_message(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{timestamp} | {message}"
    print(full_message)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(full_message + "\n")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_validation_error(error: str) -> str:
    question_patterns = [
        "wrong_field_name_for_question_number",
        "wrong_field_name_for_question",
        "wrong_field_name_for_options",
        "wrong_field_name_for_answer",
        "missing_question_number",
        "missing_question",
        "missing_options",
        "missing_answer",
        "ambiguous_question_number_field",
        "ambiguous_options_field",
        "ambiguous_answer_field",
        "answer_not_assessable_without_options",
        "answer_not_in_options",
        "is_not_dict",
        "empty_question",
        "empty_option",
        "empty_answer",
        "duplicate_options",
        "duplicate_question_text",
        "invalid_question_number",
    ]

    for pattern in question_patterns:
        if pattern in error:
            return pattern

    if "options_not_length_" in error:
        suffix = error.split("options_not_length_", 1)[1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if digits:
            return f"options_not_length_{digits}"
        return "options_not_length"

    if "learner_error_" in error:
        if "wrong_field_name_for_error" in error:
            return "wrong_field_name_for_error"
        if "wrong_field_name_for_explanation" in error:
            return "wrong_field_name_for_explanation"
        if "missing_error" in error:
            return "missing_error"
        if "missing_explanation" in error:
            return "missing_explanation"
        if "empty_error" in error:
            return "empty_error"
        if "empty_explanation" in error:
            return "empty_explanation"
        if "ambiguous_explanation_field" in error:
            return "ambiguous_explanation_field"
        if "is_not_dict" in error:
            return "learner_error_is_not_dict"

    if error.startswith("wrong_field_name_for_"):
        return error.split(":", 1)[0]

    if error.startswith("missing_"):
        return error

    if error.startswith("invalid_"):
        return error

    if error.startswith("unexpected_field:"):
        return "unexpected_field"

    if error.startswith("missing_question_object_for_expected_number:"):
        return "missing_question_object_for_expected_number"

    if error.startswith("validator_exception:"):
        return "validator_exception"

    return error

def compact_validation_result(item: dict) -> dict:
    compact = {
        "attempt": item.get("attempt"),
        "mode_id": item.get("mode_id"),
        "model": item.get("model"),
        "endpoint": item.get("endpoint"),
        "success": item.get("success"),
        "attempt_file": item.get("attempt_file"),
        "validation_errors": item.get("validation_errors", []),
    }

    if "strict_valid" in item:
        compact["strict_valid"] = item.get("strict_valid")

    if "tolerant_valid" in item:
        compact["tolerant_valid"] = item.get("tolerant_valid")

    if "recoverable" in item:
        compact["recoverable"] = item.get("recoverable")

    return compact

def build_strict_summary(validation_results: list[dict]) -> dict:
    total = len(validation_results)
    strict_valid_count = sum(1 for x in validation_results if x.get("strict_valid") is True)
    strict_invalid_count = total - strict_valid_count

    raw_error_counter = Counter()
    normalized_error_counter = Counter()

    for item in validation_results:
        for err in item.get("validation_errors", []):
            raw_error_counter[err] += 1
            normalized_error_counter[normalize_validation_error(err)] += 1

    return {
        "validated_attempts": total,
        "strict_valid_count": strict_valid_count,
        "strict_invalid_count": strict_invalid_count,
        "strict_valid_rate": round(strict_valid_count / total, 4) if total else 0.0,
        "validation_error_counts_raw": dict(raw_error_counter),
        "validation_error_counts_normalized": dict(normalized_error_counter)
    }


def build_tolerant_summary(validation_results: list[dict]) -> dict:
    total = len(validation_results)
    tolerant_valid_count = sum(1 for x in validation_results if x.get("tolerant_valid") is True)
    tolerant_invalid_count = total - tolerant_valid_count
    recoverable_count = sum(1 for x in validation_results if x.get("recoverable") is True)

    raw_error_counter = Counter()
    normalized_error_counter = Counter()

    fatal_raw_error_counter = Counter()
    fatal_normalized_error_counter = Counter()

    non_fatal_raw_error_counter = Counter()
    non_fatal_normalized_error_counter = Counter()

    for item in validation_results:
        for err in item.get("validation_errors", []):
            raw_error_counter[err] += 1
            normalized_error_counter[normalize_validation_error(err)] += 1

        for err in item.get("fatal_validation_errors", []):
            fatal_raw_error_counter[err] += 1
            fatal_normalized_error_counter[normalize_validation_error(err)] += 1

        for err in item.get("non_fatal_validation_errors", []):
            non_fatal_raw_error_counter[err] += 1
            non_fatal_normalized_error_counter[normalize_validation_error(err)] += 1

    return {
        "validated_attempts": total,
        "tolerant_valid_count": tolerant_valid_count,
        "tolerant_invalid_count": tolerant_invalid_count,
        "tolerant_valid_rate": round(tolerant_valid_count / total, 4) if total else 0.0,
        "recoverable_count": recoverable_count,
        "recoverable_rate": round(recoverable_count / total, 4) if total else 0.0,
        "validation_error_counts_raw": dict(raw_error_counter),
        "validation_error_counts_normalized": dict(normalized_error_counter),
        "fatal_error_counts_raw": dict(fatal_raw_error_counter),
        "fatal_error_counts_normalized": dict(fatal_normalized_error_counter),
        "non_fatal_error_counts_raw": dict(non_fatal_raw_error_counter),
        "non_fatal_error_counts_normalized": dict(non_fatal_normalized_error_counter)
    }

def validate_endpoint_dir_strict(endpoint_dir: Path) -> None:
    validation_results = []

    for attempt_file in sorted(endpoint_dir.glob("attempt_*.json")):
        try:
            attempt_data = load_json(attempt_file)
            mode_id = attempt_data.get("mode_id")

            if is_test_mode(mode_id):
                validation = validate_test_strict(attempt_data)
            elif is_grammar_mode(mode_id):
                validation = validate_grammar_strict(attempt_data)
            elif is_writing_mode(mode_id):
                validation = validate_writing_strict(attempt_data)
            else:
                validation = {
                    "attempt": attempt_data.get("attempt"),
                    "mode_id": mode_id,
                    "model": attempt_data.get("model"),
                    "endpoint": attempt_data.get("endpoint"),
                    "success": attempt_data.get("success"),
                    "strict_valid": False,
                    "validation_errors": ["unsupported_mode_for_strict_validation"],
                    "checks": {}
                }

            validation["attempt_file"] = attempt_file.name
            validation_results.append(validation)

        except Exception as e:
            validation_results.append({
                "attempt_file": attempt_file.name,
                "attempt": None,
                "mode_id": endpoint_dir.parent.parent.name,
                "model": endpoint_dir.parent.name,
                "endpoint": endpoint_dir.name,
                "success": False,
                "strict_valid": False,
                "validation_errors": [f"validator_exception: {e}"],
                "checks": {}
            })

    compact_results = [compact_validation_result(x) for x in validation_results]
    save_result(compact_results, str(endpoint_dir / "validation_strict.json"))

    summary = build_strict_summary(validation_results)
    save_result(summary, str(endpoint_dir / "summary_strict.json"))

def validate_endpoint_dir_tolerant(endpoint_dir: Path) -> None:
    validation_results = []

    for attempt_file in sorted(endpoint_dir.glob("attempt_*.json")):
        try:
            attempt_data = load_json(attempt_file)
            mode_id = attempt_data.get("mode_id")

            if is_test_mode(mode_id):
                validation = validate_test_tolerant(attempt_data)
            elif is_grammar_mode(mode_id):
                validation = validate_grammar_tolerant(attempt_data)
            elif is_writing_mode(mode_id):
                validation = validate_writing_tolerant(attempt_data)
            else:
                validation = {
                    "attempt": attempt_data.get("attempt"),
                    "mode_id": mode_id,
                    "model": attempt_data.get("model"),
                    "endpoint": attempt_data.get("endpoint"),
                    "success": attempt_data.get("success"),
                    "tolerant_valid": False,
                    "recoverable": False,
                    "validation_errors": ["unsupported_mode_for_tolerant_validation"],
                    "fatal_validation_errors": [],
                    "non_fatal_validation_errors": [],
                    "checks": {}
                }

            validation["attempt_file"] = attempt_file.name
            validation_results.append(validation)

        except Exception as e:
            validation_results.append({
                "attempt_file": attempt_file.name,
                "attempt": None,
                "mode_id": endpoint_dir.parent.parent.name,
                "model": endpoint_dir.parent.name,
                "endpoint": endpoint_dir.name,
                "success": False,
                "tolerant_valid": False,
                "recoverable": False,
                "validation_errors": [f"validator_exception: {e}"],
                "fatal_validation_errors": [f"validator_exception: {e}"],
                "non_fatal_validation_errors": [],
                "checks": {}
            })

    compact_results = [compact_validation_result(x) for x in validation_results]
    save_result(compact_results, str(endpoint_dir / "validation_tolerant.json"))

    summary = build_tolerant_summary(validation_results)
    save_result(summary, str(endpoint_dir / "summary_tolerant.json"))

def main():
    root = Path("results")

    endpoint_dirs = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name in {"chat", "generate"}:
            mode_dir_name = path.parent.parent.name
            if is_test_mode(mode_dir_name) or is_grammar_mode(mode_dir_name) or is_writing_mode(mode_dir_name):
                endpoint_dirs.append(path)

    log_message("=== VALIDATION STARTED ===")

    for endpoint_dir in sorted(endpoint_dirs):
        log_message(f"VALIDATING STRICT | {endpoint_dir}")
        validate_endpoint_dir_strict(endpoint_dir)
        log_message(f"STRICT SUMMARY SAVED | {endpoint_dir}")

        log_message(f"VALIDATING TOLERANT | {endpoint_dir}")
        validate_endpoint_dir_tolerant(endpoint_dir)
        log_message(f"TOLERANT SUMMARY SAVED | {endpoint_dir}")

    log_message("=== VALIDATION FINISHED ===")


if __name__ == "__main__":
    main()