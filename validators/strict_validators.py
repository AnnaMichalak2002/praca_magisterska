from typing import Any
from validators.specs import TEST_SECTION_SPECS


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_question_strict(
    question_obj: Any,
    idx: int,
    expected_question_number: int,
) -> tuple[list[str], dict]:
    errors: list[str] = []

    checks = {
        "is_dict": False,
        "has_exact_fields": False,
        "question_number_valid": False,
        "question_non_empty": False,
        "options_is_list": False,
        "options_length_valid": False,
        "options_non_empty": False,
        "options_unique": False,
        "answer_non_empty": False,
        "answer_in_options": False,
    }

    required_fields = {"question_number", "question", "options", "answer"}

    if not isinstance(question_obj, dict):
        errors.append(f"question_{idx}_is_not_dict")
        return errors, checks

    checks["is_dict"] = True

    actual_fields = set(question_obj.keys())
    missing_fields = required_fields - actual_fields
    extra_fields = actual_fields - required_fields

    if missing_fields:
        for field in sorted(missing_fields):
            errors.append(f"question_{idx}_missing_{field}")

    if extra_fields:
        for field in sorted(extra_fields):
            errors.append(f"question_{idx}_unexpected_field:{field}")

    if not missing_fields and not extra_fields:
        checks["has_exact_fields"] = True

    question_number = question_obj.get("question_number")
    question_text = question_obj.get("question")
    options = question_obj.get("options")
    answer = question_obj.get("answer")

    if isinstance(question_number, int) and question_number == expected_question_number:
        checks["question_number_valid"] = True
    else:
        errors.append(f"question_{idx}_invalid_question_number")

    if _is_non_empty_string(question_text):
        checks["question_non_empty"] = True
    else:
        errors.append(f"question_{idx}_empty_question")

    if isinstance(options, list):
        checks["options_is_list"] = True

        if len(options) == 4:
            checks["options_length_valid"] = True
        else:
            errors.append(f"question_{idx}_options_not_length_4")

        if all(_is_non_empty_string(opt) for opt in options):
            checks["options_non_empty"] = True
        else:
            errors.append(f"question_{idx}_empty_option")

        if len(options) == len(set(options)):
            checks["options_unique"] = True
        else:
            errors.append(f"question_{idx}_duplicate_options")
    else:
        errors.append(f"question_{idx}_options_is_not_list")

    if _is_non_empty_string(answer):
        checks["answer_non_empty"] = True
    else:
        errors.append(f"question_{idx}_empty_answer")

    if isinstance(options, list) and _is_non_empty_string(answer):
        if answer in options:
            checks["answer_in_options"] = True
        else:
            errors.append(f"question_{idx}_answer_not_in_options")

    return errors, checks


def validate_test_strict(attempt_data: dict) -> dict:
    mode_id = attempt_data.get("mode_id")
    spec = TEST_SECTION_SPECS.get(mode_id)

    result = {
        "attempt": attempt_data.get("attempt"),
        "mode_id": mode_id,
        "model": attempt_data.get("model"),
        "endpoint": attempt_data.get("endpoint"),
        "success": attempt_data.get("success"),
        "strict_valid": False,
        "validation_errors": [],
        "checks": {
            "response_is_dict": False,
            "top_level_exact_fields": False,
            "exercise_type_valid": False,
            "section_id_valid": False,
            "section_title_valid": False,
            "cefr_range_valid": False,
            "questions_is_list": False,
            "question_count_valid": False,
            "question_numbering_valid": False,
            "reading_text_presence_valid": None,
            "reading_text_non_empty": None,
            "all_questions_strict_valid": False,
        },
        "question_checks": []
    }

    if spec is None:
        result["validation_errors"].append("unknown_mode_id")
        return result

    if not attempt_data.get("success"):
        result["validation_errors"].append("original_attempt_not_successful")
        return result

    response = attempt_data.get("response")
    if not isinstance(response, dict):
        result["validation_errors"].append("response_is_not_dict")
        return result

    result["checks"]["response_is_dict"] = True

    required_top_fields = {
        "exercise_type",
        "section_id",
        "section_title",
        "cefr_range",
        "questions",
    }

    if spec["requires_reading_text"]:
        required_top_fields.add("reading_text")

    actual_top_fields = set(response.keys())
    missing_top_fields = required_top_fields - actual_top_fields
    extra_top_fields = actual_top_fields - required_top_fields

    if missing_top_fields:
        for field in sorted(missing_top_fields):
            result["validation_errors"].append(f"missing_{field}")

    if extra_top_fields:
        for field in sorted(extra_top_fields):
            result["validation_errors"].append(f"unexpected_field:{field}")

    if not missing_top_fields and not extra_top_fields:
        result["checks"]["top_level_exact_fields"] = True

    if response.get("exercise_type") == "CEFR English Placement Test":
        result["checks"]["exercise_type_valid"] = True
    else:
        result["validation_errors"].append("invalid_exercise_type")

    if response.get("section_id") == spec["section_id"]:
        result["checks"]["section_id_valid"] = True
    else:
        result["validation_errors"].append("invalid_section_id")

    if response.get("section_title") == spec["section_title"]:
        result["checks"]["section_title_valid"] = True
    else:
        result["validation_errors"].append("invalid_section_title")

    if response.get("cefr_range") == spec["cefr_range"]:
        result["checks"]["cefr_range_valid"] = True
    else:
        result["validation_errors"].append("invalid_cefr_range")

    questions = response.get("questions")
    if not isinstance(questions, list):
        result["validation_errors"].append("questions_is_not_list")
        return result

    result["checks"]["questions_is_list"] = True

    if len(questions) == spec["question_count"]:
        result["checks"]["question_count_valid"] = True
    else:
        result["validation_errors"].append("invalid_question_count")

    actual_numbers: list[int] = []
    normalized_question_texts = []
    all_questions_strict_valid = True

    for idx, expected_number in enumerate(spec["question_numbers"], start=1):
        if idx <= len(questions):
            q = questions[idx - 1]
            q_errors, q_checks = _validate_question_strict(q, idx, expected_number)
            result["validation_errors"].extend(q_errors)
            result["question_checks"].append({
                "question_index": idx,
                "expected_question_number": expected_number,
                "checks": q_checks,
                "valid": len(q_errors) == 0,
            })

            if isinstance(q, dict) and isinstance(q.get("question_number"), int):
                actual_numbers.append(q["question_number"])
            
            if isinstance(q, dict):
                question_text = q.get("question")
                if isinstance(question_text, str) and question_text.strip():
                    normalized_question_texts.append(question_text.strip().lower())
            
            if q_errors:
                all_questions_strict_valid = False
        else:
            result["validation_errors"].append(f"missing_question_object_for_expected_number:{expected_number}")
            result["question_checks"].append({
                "question_index": idx,
                "expected_question_number": expected_number,
                "checks": {},
                "valid": False,
            })
            all_questions_strict_valid = False

    if actual_numbers == spec["question_numbers"]:
        result["checks"]["question_numbering_valid"] = True
    else:
        result["validation_errors"].append("invalid_question_numbering")

    if len(normalized_question_texts) != len(set(normalized_question_texts)):
        result["validation_errors"].append("duplicate_question_text")
        result["checks"]["all_questions_strict_valid"] = False
        
    if spec["requires_reading_text"]:
        result["checks"]["reading_text_presence_valid"] = "reading_text" in response
        if "reading_text" not in response:
            result["validation_errors"].append("missing_reading_text")
        elif _is_non_empty_string(response.get("reading_text")):
            result["checks"]["reading_text_non_empty"] = True
        else:
            result["checks"]["reading_text_non_empty"] = False
            result["validation_errors"].append("empty_reading_text")
    else:
        if "reading_text" in response:
            result["checks"]["reading_text_presence_valid"] = False
            result["validation_errors"].append("unexpected_reading_text")
        else:
            result["checks"]["reading_text_presence_valid"] = True
            result["checks"]["reading_text_non_empty"] = True

    result["checks"]["all_questions_strict_valid"] = all_questions_strict_valid

    result["strict_valid"] = len(result["validation_errors"]) == 0
    return result