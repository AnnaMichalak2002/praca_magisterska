from pathlib import Path
import json
from collections import Counter
from datetime import datetime

from llm_utils import save_result
from validation.specs import is_test_mode, is_grammar_mode, is_writing_mode, is_vocabulary_mode
from validation.tolerant_validators import (
    validate_test_tolerant,
    validate_grammar_tolerant,
    validate_writing_tolerant,
    validate_vocabulary_tolerant
)

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
        "wrong_field_name_for_word",
        "wrong_field_name_for_options",
        "wrong_field_name_for_answer",

        "missing_question_number",
        "missing_question",
        "missing_word",
        "missing_options",
        "missing_answer",

        "ambiguous_question_number_field",
        "ambiguous_options_field",
        "ambiguous_answer_field",

        "answer_not_assessable_without_options",
        "answer_not_in_options",

        "is_not_dict",

        "empty_question",
        "empty_word",
        "empty_option",
        "empty_answer",

        "options_is_not_list",

        "duplicate_options",
        "duplicate_question_text",
        "duplicate_word",

        "invalid_question_number",
        "word_present_in_options",

        "wrong_field_name_for_definition",
        "missing_definition",
        "empty_definition",
        "duplicate_definition",
        "definition_present_in_options",
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
    return {
        "attempt": item.get("attempt"),
        "mode_id": item.get("mode_id"),
        "model": item.get("model"),
        "endpoint": item.get("endpoint"),
        "success": item.get("success"),
        "attempt_file": item.get("attempt_file"),
        "tolerant_valid": item.get("tolerant_valid"),
        "recoverable": item.get("recoverable"),
        #"validation_errors": item.get("validation_errors", []),
        "fatal_validation_errors": item.get("fatal_validation_errors", []),
        "non_fatal_validation_errors": item.get("non_fatal_validation_errors", []),
    }


def build_tolerant_summary(validation_results: list[dict]) -> dict:
    all_attempts = len(validation_results)

    not_validated_count = sum(
        1
        for x in validation_results
        if "original_attempt_not_successful" in x.get("validation_errors", [])
    )

    validated_items = [
        x for x in validation_results
        if "original_attempt_not_successful" not in x.get("validation_errors", [])
    ]

    validated_attempts = len(validated_items)

    strictly_valid_count = sum(
        1 for x in validated_items
        if x.get("tolerant_valid") is True and x.get("recoverable") is False
    )

    recovered_by_tolerant_count = sum(
        1 for x in validated_items
        if x.get("tolerant_valid") is True and x.get("recoverable") is True
    )

    final_valid_count = sum(
        1 for x in validated_items
        if x.get("tolerant_valid") is True
    )

    final_invalid_count = validated_attempts - final_valid_count
    non_strict_count = validated_attempts - strictly_valid_count

    raw_error_counter = Counter()
    normalized_error_counter = Counter()

    fatal_raw_error_counter = Counter()
    fatal_normalized_error_counter = Counter()

    non_fatal_raw_error_counter = Counter()
    non_fatal_normalized_error_counter = Counter()

    for item in validated_items:
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
        "all_attempts": all_attempts,
        "validated_attempts": validated_attempts,
        "not_validated_attempts": not_validated_count,
        "not_validated_reasons": {
            "original_attempt_not_successful": not_validated_count
        },

        "strictly_valid_count": strictly_valid_count,
        "strictly_valid_rate": round(strictly_valid_count / validated_attempts, 4) if validated_attempts else 0.0,
        "strictly_valid_rate_over_all_attempts": round(strictly_valid_count / all_attempts, 4) if all_attempts else 0.0,

        "recovered_by_tolerant_count": recovered_by_tolerant_count,
        "recovered_by_tolerant_rate_over_validated": round(recovered_by_tolerant_count / validated_attempts, 4) if validated_attempts else 0.0,
        "recovered_by_tolerant_rate_over_non_strict": (
            round(recovered_by_tolerant_count / non_strict_count, 4)
            if non_strict_count else 0.0
        ),
        "recovered_by_tolerant_rate_over_all_attempts": (
            round(recovered_by_tolerant_count / all_attempts, 4)
            if all_attempts else 0.0
        ),

        "final_valid_count": final_valid_count,
        "final_valid_rate": round(final_valid_count / validated_attempts, 4) if validated_attempts else 0.0,
        "final_valid_rate_over_all_attempts": round(final_valid_count / all_attempts, 4) if all_attempts else 0.0,

        "final_invalid_count": final_invalid_count,
        "final_invalid_rate": round(final_invalid_count / validated_attempts, 4) if validated_attempts else 0.0,
        "final_invalid_rate_over_all_attempts": round(final_invalid_count / all_attempts, 4) if all_attempts else 0.0,
        
        "fatal_error_counts_normalized": dict(fatal_normalized_error_counter),
        "non_fatal_error_counts_normalized": dict(non_fatal_normalized_error_counter),
    }


def validate_result_dir_tolerant(result_dir: Path) -> None:
    validation_results = []

    for attempt_file in sorted(result_dir.glob("attempt_*.json")):
        try:
            attempt_data = load_json(attempt_file)
            mode_id = attempt_data.get("mode_id")

            if is_test_mode(mode_id):
                validation = validate_test_tolerant(attempt_data)
            elif is_grammar_mode(mode_id):
                validation = validate_grammar_tolerant(attempt_data)
            elif is_writing_mode(mode_id):
                validation = validate_writing_tolerant(attempt_data)
            elif is_vocabulary_mode(mode_id):
                validation = validate_vocabulary_tolerant(attempt_data)
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
                    "fatal_validation_errors": ["unsupported_mode_for_tolerant_validation"],
                    "non_fatal_validation_errors": [],
                    "checks": {},
                }

            validation["attempt_file"] = attempt_file.name
            validation_results.append(validation)

        except Exception as e:
            validation_results.append({
                "attempt_file": attempt_file.name,
                "attempt": None,
                "mode_id": result_dir.parent.name,
                "model": result_dir.name,
                "endpoint": "chat",
                "success": False,
                "tolerant_valid": False,
                "recoverable": False,
                "validation_errors": [f"validator_exception: {e}"],
                "fatal_validation_errors": [f"validator_exception: {e}"],
                "non_fatal_validation_errors": [],
                "checks": {},
            })

    compact_results = [compact_validation_result(x) for x in validation_results]
    save_result(compact_results, str(result_dir / "validation_tolerant.json"))

    summary = build_tolerant_summary(validation_results)
    save_result(summary, str(result_dir / "summary_tolerant.json"))


def main():
    root = Path("results")

    result_dirs = []
    for path in root.rglob("*"):
        if path.is_dir() and any(path.glob("attempt_*.json")):
            mode_dir_name = path.parent.name
            if (
                is_test_mode(mode_dir_name)
                or is_grammar_mode(mode_dir_name)
                or is_writing_mode(mode_dir_name)
                or is_vocabulary_mode(mode_dir_name)
            ):
                result_dirs.append(path)

    log_message("=== TOLERANT VALIDATION STARTED ===")

    for result_dir in sorted(result_dirs):
        log_message(f"VALIDATING TOLERANT | {result_dir}")
        validate_result_dir_tolerant(result_dir)
        log_message(f"TOLERANT SUMMARY SAVED | {result_dir}")

    log_message("=== TOLERANT VALIDATION FINISHED ===")


if __name__ == "__main__":
    main()