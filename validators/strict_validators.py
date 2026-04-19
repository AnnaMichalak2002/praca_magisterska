from typing import Any
from validators.specs import TEST_SECTION_SPECS, GRAMMAR_MODE_SPECS, WRITING_MODE_SPECS


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
        if "reading_text" in response:
            if _is_non_empty_string(response.get("reading_text")):
                result["checks"]["reading_text_non_empty"] = True
            else:
                result["checks"]["reading_text_non_empty"] = False
                result["validation_errors"].append("empty_reading_text")
        else:
            result["checks"]["reading_text_non_empty"] = False
    else:
        if "reading_text" in response:
            result["checks"]["reading_text_presence_valid"] = False
            result["checks"]["reading_text_non_empty"] = _is_non_empty_string(response.get("reading_text"))
        else:
            result["checks"]["reading_text_presence_valid"] = True
            result["checks"]["reading_text_non_empty"] = True

    result["checks"]["all_questions_strict_valid"] = all_questions_strict_valid

    result["strict_valid"] = len(result["validation_errors"]) == 0
    return result


def validate_grammar_strict(attempt_data: dict) -> dict:
    mode_id = attempt_data.get("mode_id")
    spec = GRAMMAR_MODE_SPECS.get(mode_id)

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
            "grammar_topic_valid": False,
            "level_valid": False,
            "questions_is_list": False,
            "question_count_valid": False,
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

    expected_top_level = {
        "exercise_type",
        "grammar_topic",
        "level",
        "questions",
    }

    actual_top_level = set(response.keys())
    unexpected_top = actual_top_level - expected_top_level
    missing_top = expected_top_level - actual_top_level

    if not missing_top and not unexpected_top:
        result["checks"]["top_level_exact_fields"] = True
    else:
        for field in sorted(missing_top):
            result["validation_errors"].append(f"missing_{field}")
        for field in sorted(unexpected_top):
            result["validation_errors"].append(f"unexpected_field:{field}")

    if response.get("exercise_type") == spec["exercise_type"]:
        result["checks"]["exercise_type_valid"] = True
    else:
        result["validation_errors"].append("invalid_exercise_type")

    if response.get("grammar_topic") == spec["grammar_topic"]:
        result["checks"]["grammar_topic_valid"] = True
    else:
        result["validation_errors"].append("invalid_grammar_topic")

    if response.get("level") == spec["level"]:
        result["checks"]["level_valid"] = True
    else:
        result["validation_errors"].append("invalid_level")

    questions = response.get("questions")
    if not isinstance(questions, list):
        result["validation_errors"].append("questions_is_not_list")
        return result

    result["checks"]["questions_is_list"] = True

    if len(questions) == spec["question_count"]:
        result["checks"]["question_count_valid"] = True
    else:
        result["validation_errors"].append("invalid_question_count")

    normalized_question_texts = []
    all_questions_strict_valid = True

    for idx, q in enumerate(questions, start=1):
        q_result = {
            "question_index": idx,
            "checks": {
                "is_dict": False,
                "has_exact_fields": False,
                "question_non_empty": False,
                "options_is_list": False,
                "options_length_valid": False,
                "options_non_empty": False,
                "options_unique": False,
                "answer_non_empty": False,
                "answer_in_options": False,
            },
            "valid": False
        }

        if not isinstance(q, dict):
            result["validation_errors"].append(f"question_{idx}_is_not_dict")
            result["question_checks"].append(q_result)
            all_questions_strict_valid = False
            continue

        q_result["checks"]["is_dict"] = True

        expected_q_fields = {"question", "options", "answer"}
        actual_q_fields = set(q.keys())

        missing_q_fields = expected_q_fields - actual_q_fields
        unexpected_q_fields = actual_q_fields - expected_q_fields

        if not missing_q_fields and not unexpected_q_fields:
            q_result["checks"]["has_exact_fields"] = True
        else:
            for field in sorted(missing_q_fields):
                result["validation_errors"].append(f"question_{idx}_missing_{field}")
            for field in sorted(unexpected_q_fields):
                result["validation_errors"].append(f"question_{idx}_unexpected_field:{field}")

        question_text = q.get("question")
        if _is_non_empty_string(question_text):
            q_result["checks"]["question_non_empty"] = True
            normalized_question_texts.append(question_text.strip().lower())
        else:
            result["validation_errors"].append(f"question_{idx}_empty_question")

        options = q.get("options")
        if isinstance(options, list):
            q_result["checks"]["options_is_list"] = True

            if len(options) == spec["options_count"]:
                q_result["checks"]["options_length_valid"] = True
            else:
                result["validation_errors"].append(f"question_{idx}_options_not_length_{spec['options_count']}")

            if all(_is_non_empty_string(opt) for opt in options):
                q_result["checks"]["options_non_empty"] = True
            else:
                result["validation_errors"].append(f"question_{idx}_empty_option")

            normalized_options = [opt.strip() for opt in options if isinstance(opt, str)]
            if len(normalized_options) == len(set(normalized_options)) and len(normalized_options) == len(options):
                q_result["checks"]["options_unique"] = True
            else:
                result["validation_errors"].append(f"question_{idx}_duplicate_options")
        else:
            result["validation_errors"].append(f"question_{idx}_options_is_not_list")
            options = None

        answer = q.get("answer")
        if _is_non_empty_string(answer):
            q_result["checks"]["answer_non_empty"] = True

            if isinstance(options, list):
                if answer in options:
                    q_result["checks"]["answer_in_options"] = True
                else:
                    result["validation_errors"].append(f"question_{idx}_answer_not_in_options")
        else:
            result["validation_errors"].append(f"question_{idx}_empty_answer")

        q_result["valid"] = all(q_result["checks"].values())
        if not q_result["valid"]:
            all_questions_strict_valid = False

        result["question_checks"].append(q_result)

    if len(normalized_question_texts) != len(set(normalized_question_texts)):
        result["validation_errors"].append("duplicate_question_text")
        all_questions_strict_valid = False

    result["checks"]["all_questions_strict_valid"] = all_questions_strict_valid
    result["strict_valid"] = len(result["validation_errors"]) == 0
    return result


def validate_writing_strict(attempt_data: dict) -> dict:
    mode_id = attempt_data.get("mode_id")
    spec = WRITING_MODE_SPECS.get(mode_id)

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
            "learner_native_language_valid": False,
            "task_content_valid": False,
            "original_text_valid": False,
            "corrected_text_non_empty": False,
            "learner_errors_is_list": False,
            "content_compliance_valid": False,
            "content_feedback_non_empty": False,
            "all_learner_errors_strict_valid": True,
        },
        "learner_error_checks": []
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

    expected_top_level = {
        "exercise_type",
        "learner_native_language",
        "task_content",
        "original_text",
        "corrected_text",
        "learner_errors",
        "content_compliance",
        "content_feedback",
    }

    actual_top_level = set(response.keys())
    missing_top = expected_top_level - actual_top_level
    unexpected_top = actual_top_level - expected_top_level

    if missing_top:
        for field in sorted(missing_top):
            result["validation_errors"].append(f"missing_{field}")

    if unexpected_top:
        for field in sorted(unexpected_top):
            result["validation_errors"].append(f"unexpected_field:{field}")

    if not missing_top and not unexpected_top:
        result["checks"]["top_level_exact_fields"] = True

    if response.get("exercise_type") == spec["exercise_type"]:
        result["checks"]["exercise_type_valid"] = True
    else:
        result["validation_errors"].append("invalid_exercise_type")

    if response.get("learner_native_language") == spec["learner_native_language"]:
        result["checks"]["learner_native_language_valid"] = True
    else:
        result["validation_errors"].append("invalid_learner_native_language")

    if response.get("task_content") == spec["task_content"]:
        result["checks"]["task_content_valid"] = True
    else:
        result["validation_errors"].append("invalid_task_content")

    if response.get("original_text") == spec["original_text"]:
        result["checks"]["original_text_valid"] = True
    else:
        result["validation_errors"].append("invalid_original_text")

    if _is_non_empty_string(response.get("corrected_text")):
        result["checks"]["corrected_text_non_empty"] = True
    else:
        result["validation_errors"].append("empty_corrected_text")

    learner_errors = response.get("learner_errors")
    if isinstance(learner_errors, list):
        result["checks"]["learner_errors_is_list"] = True
    else:
        result["validation_errors"].append("learner_errors_is_not_list")
        learner_errors = None

    if response.get("content_compliance") in spec["allowed_content_compliance"]:
        result["checks"]["content_compliance_valid"] = True
    else:
        result["validation_errors"].append("invalid_content_compliance")

    if _is_non_empty_string(response.get("content_feedback")):
        result["checks"]["content_feedback_non_empty"] = True
    else:
        result["validation_errors"].append("empty_content_feedback")

    all_learner_errors_strict_valid = True

    if isinstance(learner_errors, list):
        for idx, item in enumerate(learner_errors, start=1):
            item_result = {
                "learner_error_index": idx,
                "checks": {
                    "is_dict": False,
                    "has_exact_fields": False,
                    "error_non_empty": False,
                    "explanation_non_empty": False,
                },
                "valid": False
            }

            if not isinstance(item, dict):
                result["validation_errors"].append(f"learner_error_{idx}_is_not_dict")
                result["learner_error_checks"].append(item_result)
                all_learner_errors_strict_valid = False
                continue

            item_result["checks"]["is_dict"] = True

            expected_item_fields = {"error", "explanation"}
            actual_item_fields = set(item.keys())

            missing_item_fields = expected_item_fields - actual_item_fields
            unexpected_item_fields = actual_item_fields - expected_item_fields

            if missing_item_fields:
                for field in sorted(missing_item_fields):
                    result["validation_errors"].append(f"learner_error_{idx}_missing_{field}")

            if unexpected_item_fields:
                for field in sorted(unexpected_item_fields):
                    result["validation_errors"].append(f"learner_error_{idx}_unexpected_field:{field}")

            if not missing_item_fields and not unexpected_item_fields:
                item_result["checks"]["has_exact_fields"] = True

            if _is_non_empty_string(item.get("error")):
                item_result["checks"]["error_non_empty"] = True
            else:
                result["validation_errors"].append(f"learner_error_{idx}_empty_error")

            if _is_non_empty_string(item.get("explanation")):
                item_result["checks"]["explanation_non_empty"] = True
            else:
                result["validation_errors"].append(f"learner_error_{idx}_empty_explanation")

            item_result["valid"] = all(item_result["checks"].values())
            if not item_result["valid"]:
                all_learner_errors_strict_valid = False

            result["learner_error_checks"].append(item_result)

    result["checks"]["all_learner_errors_strict_valid"] = all_learner_errors_strict_valid
    result["strict_valid"] = len(result["validation_errors"]) == 0
    return result