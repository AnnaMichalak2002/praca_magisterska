from typing import Any
from difflib import SequenceMatcher

from validators.specs import TEST_SECTION_SPECS, GRAMMAR_MODE_SPECS 


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _infer_top_level_field(response: dict, canonical_name: str):
    """
    Returns:
    - value
    - status: exact / inferred:<actual_key> / missing
    """
    if canonical_name in response:
        return response[canonical_name], "exact"

    canonical_compact = canonical_name.replace("_", "").lower()

    best_key = None
    best_value = None
    best_score = 0.0

    for key, value in response.items():
        key_compact = key.replace("_", "").lower()
        score = SequenceMatcher(None, canonical_compact, key_compact).ratio()

        if score > best_score:
            best_score = score
            best_key = key
            best_value = value

    if best_score >= 0.72:
        return best_value, f"inferred:{best_key}"

    return None, "missing"


def _infer_question_fields(question_obj: dict, idx: int, validation_errors: list[str]) -> dict:
    extracted = {
        "question_number": None,
        "question": None,
        "options": None,
        "answer": None
    }

    used_keys = set()

    # exact canonical fields first
    if "question_number" in question_obj:
        extracted["question_number"] = question_obj["question_number"]
        used_keys.add("question_number")

    if "question" in question_obj:
        extracted["question"] = question_obj["question"]
        used_keys.add("question")

    if "options" in question_obj:
        extracted["options"] = question_obj["options"]
        used_keys.add("options")

    if "answer" in question_obj:
        extracted["answer"] = question_obj["answer"]
        used_keys.add("answer")

    # infer question_number
    if extracted["question_number"] is None:
        int_candidates = [(k, v) for k, v in question_obj.items() if isinstance(v, int)]
        if len(int_candidates) == 1:
            key, value = int_candidates[0]
            extracted["question_number"] = value
            used_keys.add(key)
            validation_errors.append(f"question_{idx}_wrong_field_name_for_question_number:{key}")
        elif len(int_candidates) == 0:
            validation_errors.append(f"question_{idx}_missing_question_number")
        else:
            validation_errors.append(f"question_{idx}_ambiguous_question_number_field")

    # infer options
    if extracted["options"] is None:
        list_candidates = [
            (k, v) for k, v in question_obj.items()
            if isinstance(v, list) and all(isinstance(x, str) for x in v)
        ]
        if len(list_candidates) == 1:
            key, value = list_candidates[0]
            extracted["options"] = value
            used_keys.add(key)
            validation_errors.append(f"question_{idx}_wrong_field_name_for_options:{key}")
        elif len(list_candidates) == 0:
            validation_errors.append(f"question_{idx}_missing_options")
        else:
            validation_errors.append(f"question_{idx}_ambiguous_options_field")

    # remaining strings
    remaining_string_candidates = [
        (k, v) for k, v in question_obj.items()
        if isinstance(v, str) and k not in used_keys
    ]

    # infer question as longest remaining string
    if extracted["question"] is None:
        if remaining_string_candidates:
            q_key, q_value = max(remaining_string_candidates, key=lambda item: len(item[1]))
            extracted["question"] = q_value
            used_keys.add(q_key)
            validation_errors.append(f"question_{idx}_wrong_field_name_for_question:{q_key}")
        else:
            validation_errors.append(f"question_{idx}_missing_question")

    remaining_string_candidates = [
        (k, v) for k, v in question_obj.items()
        if isinstance(v, str) and k not in used_keys
    ]

    # infer answer
    if extracted["answer"] is None:
        if len(remaining_string_candidates) == 1:
            a_key, a_value = remaining_string_candidates[0]
            extracted["answer"] = a_value
            used_keys.add(a_key)
            validation_errors.append(f"question_{idx}_wrong_field_name_for_answer:{a_key}")
        elif len(remaining_string_candidates) == 0:
            validation_errors.append(f"question_{idx}_missing_answer")
        else:
            if extracted["options"] is not None:
                matching_candidates = [
                    (k, v) for k, v in remaining_string_candidates
                    if v in extracted["options"]
                ]
                if len(matching_candidates) == 1:
                    a_key, a_value = matching_candidates[0]
                    extracted["answer"] = a_value
                    used_keys.add(a_key)
                    validation_errors.append(f"question_{idx}_wrong_field_name_for_answer:{a_key}")
                else:
                    validation_errors.append(f"question_{idx}_ambiguous_answer_field")
            else:
                validation_errors.append(f"question_{idx}_ambiguous_answer_field")

    return extracted


def _validate_question_tolerant(
    question_obj: Any,
    idx: int,
) -> tuple[list[str], dict, dict]:
    errors: list[str] = []

    checks = {
        "is_dict": False,
        "question_number_present": False,
        "question_non_empty": False,
        "options_is_list": False,
        "options_length_valid": False,
        "options_non_empty": False,
        "options_unique": False,
        "answer_present": False,
        "answer_non_empty": False,
        "answer_in_options": False,
    }

    if not isinstance(question_obj, dict):
        errors.append(f"question_{idx}_is_not_dict")
        return errors, checks, {
            "question_number": None,
            "question": None,
            "options": None,
            "answer": None
        }

    checks["is_dict"] = True

    extracted = _infer_question_fields(question_obj, idx, errors)

    question_number = extracted["question_number"]
    question_text = extracted["question"]
    options = extracted["options"]
    answer = extracted["answer"]

    if isinstance(question_number, int):
        checks["question_number_present"] = True

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

    if answer is not None:
        checks["answer_present"] = True

    if _is_non_empty_string(answer):
        checks["answer_non_empty"] = True
    else:
        errors.append(f"question_{idx}_empty_answer")

    if isinstance(options, list) and _is_non_empty_string(answer):
        if answer in options:
            checks["answer_in_options"] = True
        else:
            errors.append(f"question_{idx}_answer_not_in_options")

    return errors, checks, extracted

def _is_non_fatal_tolerant_error(error: str) -> bool:
    return (
        error.startswith("wrong_field_name_for_")
        or "_wrong_field_name_for_" in error
    )

def validate_test_tolerant(attempt_data: dict) -> dict:
    mode_id = attempt_data.get("mode_id")
    spec = TEST_SECTION_SPECS.get(mode_id)

    result = {
        "attempt": attempt_data.get("attempt"),
        "mode_id": mode_id,
        "model": attempt_data.get("model"),
        "endpoint": attempt_data.get("endpoint"),
        "success": attempt_data.get("success"),
        "tolerant_valid": False,
        "recoverable": False,
        "validation_errors": [],
        "fatal_validation_errors": [],
        "non_fatal_validation_errors": [],
        "checks": {
            "response_is_dict": False,
            "exercise_type_present_or_inferred": False,
            "exercise_type_valid": False,
            "section_id_present_or_inferred": False,
            "section_id_valid": False,
            "section_title_present_or_inferred": False,
            "section_title_valid": False,
            "cefr_range_present_or_inferred": False,
            "cefr_range_valid": False,
            "questions_is_list": False,
            "question_count_valid": False,
            "question_numbering_valid": None,
            "reading_text_valid": None,
            "all_questions_tolerant_valid": False,
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

    # top-level tolerant checks
    exercise_type, ex_status = _infer_top_level_field(response, "exercise_type")
    if ex_status == "missing":
        result["validation_errors"].append("missing_exercise_type")
    else:
        result["checks"]["exercise_type_present_or_inferred"] = True
        if ex_status.startswith("inferred:"):
            result["validation_errors"].append(
                f"wrong_field_name_for_exercise_type:{ex_status.split(':', 1)[1]}"
            )
        if exercise_type == "CEFR English Placement Test":
            result["checks"]["exercise_type_valid"] = True
        else:
            result["validation_errors"].append("invalid_exercise_type")

    section_id, sid_status = _infer_top_level_field(response, "section_id")
    if sid_status == "missing":
        result["validation_errors"].append("missing_section_id")
    else:
        result["checks"]["section_id_present_or_inferred"] = True
        if sid_status.startswith("inferred:"):
            result["validation_errors"].append(
                f"wrong_field_name_for_section_id:{sid_status.split(':', 1)[1]}"
            )
        if section_id == spec["section_id"]:
            result["checks"]["section_id_valid"] = True
        else:
            result["validation_errors"].append("invalid_section_id")

    section_title, st_status = _infer_top_level_field(response, "section_title")
    if st_status == "missing":
        result["validation_errors"].append("missing_section_title")
    else:
        result["checks"]["section_title_present_or_inferred"] = True
        if st_status.startswith("inferred:"):
            result["validation_errors"].append(
                f"wrong_field_name_for_section_title:{st_status.split(':', 1)[1]}"
            )
        if section_title == spec["section_title"]:
            result["checks"]["section_title_valid"] = True
        else:
            result["validation_errors"].append("invalid_section_title")

    cefr_range, cr_status = _infer_top_level_field(response, "cefr_range")
    if cr_status == "missing":
        result["validation_errors"].append("missing_cefr_range")
    else:
        result["checks"]["cefr_range_present_or_inferred"] = True
        if cr_status.startswith("inferred:"):
            result["validation_errors"].append(
                f"wrong_field_name_for_cefr_range:{cr_status.split(':', 1)[1]}"
            )
        if cefr_range == spec["cefr_range"]:
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
    numbering_assessable = True
    all_questions_tolerant_valid = True

    for idx, q in enumerate(questions, start=1):
        q_errors, q_checks, extracted = _validate_question_tolerant(q, idx)
        result["validation_errors"].extend(q_errors)

        result["question_checks"].append({
            "question_index": idx,
            "checks": q_checks,
            "valid": len(q_errors) == 0,
            "extracted_fields": extracted
        })

        if isinstance(extracted["question_number"], int):
            actual_numbers.append(extracted["question_number"])
        else:
            numbering_assessable = False

        question_text = extracted.get("question")
        if isinstance(question_text, str) and question_text.strip():
            normalized_question_texts.append(question_text.strip().lower())

        if q_errors:
            all_questions_tolerant_valid = False

    result["checks"]["all_questions_tolerant_valid"] = all_questions_tolerant_valid

    if numbering_assessable:
        if actual_numbers == spec["question_numbers"]:
            result["checks"]["question_numbering_valid"] = True
        else:
            result["checks"]["question_numbering_valid"] = False
            result["validation_errors"].append("invalid_question_numbering")
    else:
        result["checks"]["question_numbering_valid"] = None

    if len(normalized_question_texts) != len(set(normalized_question_texts)):
        result["validation_errors"].append("duplicate_question_text")
        result["checks"]["all_questions_tolerant_valid"] = False

    requires_reading_text = spec["requires_reading_text"]

    if requires_reading_text:
        reading_text, rt_status = _infer_top_level_field(response, "reading_text")
        if rt_status == "missing":
            result["checks"]["reading_text_valid"] = False
            result["validation_errors"].append("missing_reading_text")
        else:
            if rt_status.startswith("inferred:"):
                result["validation_errors"].append(
                    f"wrong_field_name_for_reading_text:{rt_status.split(':', 1)[1]}"
                )
            if _is_non_empty_string(reading_text):
                result["checks"]["reading_text_valid"] = True
            else:
                result["checks"]["reading_text_valid"] = False
                result["validation_errors"].append("empty_reading_text")
    else:
        if "reading_text" in response and response.get("reading_text"):
            result["checks"]["reading_text_valid"] = False
            result["validation_errors"].append("unexpected_reading_text")
        else:
            result["checks"]["reading_text_valid"] = True

    fatal_errors = [
        err for err in result["validation_errors"]
        if not _is_non_fatal_tolerant_error(err)
    ]

    non_fatal_errors = [
        err for err in result["validation_errors"]
        if _is_non_fatal_tolerant_error(err)
    ]

    result["fatal_validation_errors"] = fatal_errors
    result["non_fatal_validation_errors"] = non_fatal_errors

    result["tolerant_valid"] = len(fatal_errors) == 0
    result["recoverable"] = len(fatal_errors) == 0 and len(non_fatal_errors) > 0

    return result

def _infer_grammar_question_fields_tolerant(question_obj: dict, idx: int, validation_errors: list[str]) -> dict:
    """
    Tolerant inference for grammar questions:
    expected logical fields:
    - question -> non-empty string
    - options -> list[str]
    - answer -> string
    """
    extracted = {
        "question": None,
        "options": None,
        "answer": None,
    }

    used_keys = set()

    # exact canonical fields first
    if "question" in question_obj:
        extracted["question"] = question_obj["question"]
        used_keys.add("question")

    if "options" in question_obj:
        extracted["options"] = question_obj["options"]
        used_keys.add("options")

    if "answer" in question_obj:
        extracted["answer"] = question_obj["answer"]
        used_keys.add("answer")

    # infer options
    if extracted["options"] is None:
        list_candidates = [
            (k, v) for k, v in question_obj.items()
            if isinstance(v, list) and all(isinstance(x, str) for x in v)
        ]
        if len(list_candidates) == 1:
            key, value = list_candidates[0]
            extracted["options"] = value
            used_keys.add(key)
            validation_errors.append(f"question_{idx}_wrong_field_name_for_options:{key}")
        elif len(list_candidates) == 0:
            validation_errors.append(f"question_{idx}_missing_options")
        else:
            validation_errors.append(f"question_{idx}_ambiguous_options_field")

    # remaining string fields
    remaining_string_candidates = [
        (k, v) for k, v in question_obj.items()
        if isinstance(v, str) and k not in used_keys
    ]

    # infer question as longest remaining string
    if extracted["question"] is None:
        if remaining_string_candidates:
            q_key, q_value = max(remaining_string_candidates, key=lambda item: len(item[1]))
            extracted["question"] = q_value
            used_keys.add(q_key)
            validation_errors.append(f"question_{idx}_wrong_field_name_for_question:{q_key}")
        else:
            validation_errors.append(f"question_{idx}_missing_question")

    # recompute remaining strings after taking question
    remaining_string_candidates = [
        (k, v) for k, v in question_obj.items()
        if isinstance(v, str) and k not in used_keys
    ]

    # infer answer
    if extracted["answer"] is None:
        if len(remaining_string_candidates) == 1:
            a_key, a_value = remaining_string_candidates[0]
            extracted["answer"] = a_value
            used_keys.add(a_key)
            validation_errors.append(f"question_{idx}_wrong_field_name_for_answer:{a_key}")
        elif len(remaining_string_candidates) == 0:
            validation_errors.append(f"question_{idx}_missing_answer")
        else:
            if extracted["options"] is not None:
                matching_candidates = [
                    (k, v) for k, v in remaining_string_candidates
                    if v in extracted["options"]
                ]
                if len(matching_candidates) == 1:
                    a_key, a_value = matching_candidates[0]
                    extracted["answer"] = a_value
                    used_keys.add(a_key)
                    validation_errors.append(f"question_{idx}_wrong_field_name_for_answer:{a_key}")
                elif len(matching_candidates) == 0:
                    validation_errors.append(f"question_{idx}_missing_answer")
                else:
                    validation_errors.append(f"question_{idx}_ambiguous_answer_field")
            else:
                validation_errors.append(f"question_{idx}_ambiguous_answer_field")

    return extracted

def validate_grammar_tolerant(attempt_data: dict) -> dict:
    mode_id = attempt_data.get("mode_id")
    spec = GRAMMAR_MODE_SPECS.get(mode_id)

    result = {
        "attempt": attempt_data.get("attempt"),
        "mode_id": mode_id,
        "model": attempt_data.get("model"),
        "endpoint": attempt_data.get("endpoint"),
        "success": attempt_data.get("success"),
        "tolerant_valid": False,
        "recoverable": False,
        "validation_errors": [],
        "fatal_validation_errors": [],
        "non_fatal_validation_errors": [],
        "checks": {
            "response_is_dict": False,
            "exercise_type_valid": False,
            "grammar_topic_valid": False,
            "level_valid": False,
            "questions_is_list": False,
            "question_count_valid": False,
            "all_questions_have_required_fields": False,
            "all_questions_non_empty": None,
            "all_questions_unique": None,
            "all_questions_have_3_options": None,
            "all_options_non_empty": None,
            "all_options_unique": None,
            "all_answers_non_empty": None,
            "all_answers_in_options": None,
        },
        "question_checks": []
    }

    if spec is None:
        result["fatal_validation_errors"].append("unknown_mode_id")
        result["validation_errors"] = (
            result["fatal_validation_errors"] + result["non_fatal_validation_errors"]
        )
        return result

    if not attempt_data.get("success"):
        result["fatal_validation_errors"].append("original_attempt_not_successful")
        result["validation_errors"] = (
            result["fatal_validation_errors"] + result["non_fatal_validation_errors"]
        )
        return result

    response = attempt_data.get("response")
    if not isinstance(response, dict):
        result["fatal_validation_errors"].append("response_is_not_dict")
        result["validation_errors"] = (
            result["fatal_validation_errors"] + result["non_fatal_validation_errors"]
        )
        return result

    result["checks"]["response_is_dict"] = True

    # tolerant top-level field inference
    exercise_type, ex_status = _infer_top_level_field(response, "exercise_type")
    if ex_status == "missing":
        result["fatal_validation_errors"].append("missing_exercise_type")
    else:
        if ex_status.startswith("inferred:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_exercise_type:{ex_status.split(':', 1)[1]}"
            )
        if exercise_type == spec["exercise_type"]:
            result["checks"]["exercise_type_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_exercise_type")

    grammar_topic, gt_status = _infer_top_level_field(response, "grammar_topic")
    if gt_status == "missing":
        result["fatal_validation_errors"].append("missing_grammar_topic")
    else:
        if gt_status.startswith("inferred:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_grammar_topic:{gt_status.split(':', 1)[1]}"
            )
        if grammar_topic == spec["grammar_topic"]:
            result["checks"]["grammar_topic_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_grammar_topic")

    level, lv_status = _infer_top_level_field(response, "level")
    if lv_status == "missing":
        result["fatal_validation_errors"].append("missing_level")
    else:
        if lv_status.startswith("inferred:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_level:{lv_status.split(':', 1)[1]}"
            )
        if level == spec["level"]:
            result["checks"]["level_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_level")

    # questions should still be under exact name "questions"
    questions = response.get("questions")
    if not isinstance(questions, list):
        result["fatal_validation_errors"].append("questions_is_not_list")
        result["validation_errors"] = (
            result["fatal_validation_errors"] + result["non_fatal_validation_errors"]
        )
        return result

    result["checks"]["questions_is_list"] = True

    if len(questions) == spec["question_count"]:
        result["checks"]["question_count_valid"] = True
    else:
        result["fatal_validation_errors"].append("invalid_question_count")

    all_required_fields = True
    all_questions_non_empty = True
    all_questions_have_3_options = True
    all_options_non_empty = True
    all_options_unique = True
    all_answers_non_empty = True
    all_answers_in_options = True

    normalized_question_texts = []

    for idx, q in enumerate(questions, start=1):
        q_result = {
            "question_index": idx,
            "checks": {
                "is_dict": False,
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
            result["fatal_validation_errors"].append(f"question_{idx}_is_not_dict")
            result["question_checks"].append(q_result)
            all_required_fields = False
            all_questions_non_empty = False
            all_questions_have_3_options = False
            all_options_non_empty = False
            all_options_unique = False
            all_answers_non_empty = False
            all_answers_in_options = False
            continue

        q_result["checks"]["is_dict"] = True

        extracted = _infer_grammar_question_fields_tolerant(
            q,
            idx,
            result["non_fatal_validation_errors"]
        )

        question_text = extracted.get("question")
        options = extracted.get("options")
        answer = extracted.get("answer")

        if question_text is None or options is None or answer is None:
            all_required_fields = False

        if _is_non_empty_string(question_text):
            q_result["checks"]["question_non_empty"] = True
            normalized_question_texts.append(question_text.strip().lower())
        else:
            result["fatal_validation_errors"].append(f"question_{idx}_empty_question")
            all_questions_non_empty = False

        if isinstance(options, list):
            q_result["checks"]["options_is_list"] = True

            if len(options) == spec["options_count"]:
                q_result["checks"]["options_length_valid"] = True
            else:
                result["fatal_validation_errors"].append(
                    f"question_{idx}_options_not_length_{spec['options_count']}"
                )
                all_questions_have_3_options = False

            if all(_is_non_empty_string(opt) for opt in options):
                q_result["checks"]["options_non_empty"] = True
            else:
                result["fatal_validation_errors"].append(f"question_{idx}_empty_option")
                all_options_non_empty = False

            normalized_options = [opt.strip() for opt in options if isinstance(opt, str)]
            if len(normalized_options) == len(set(normalized_options)) and len(normalized_options) == len(options):
                q_result["checks"]["options_unique"] = True
            else:
                result["fatal_validation_errors"].append(f"question_{idx}_duplicate_options")
                all_options_unique = False
        else:
            result["fatal_validation_errors"].append(f"question_{idx}_options_is_not_list")
            all_questions_have_3_options = False
            all_options_non_empty = False
            all_options_unique = False
            options = None

        if _is_non_empty_string(answer):
            q_result["checks"]["answer_non_empty"] = True
        else:
            result["fatal_validation_errors"].append(f"question_{idx}_empty_answer")
            all_answers_non_empty = False

        if isinstance(options, list) and _is_non_empty_string(answer):
            if answer in options:
                q_result["checks"]["answer_in_options"] = True
            else:
                result["fatal_validation_errors"].append(f"question_{idx}_answer_not_in_options")
                all_answers_in_options = False
        elif isinstance(options, list):
            # answer empty already counted above
            all_answers_in_options = False
        else:
            all_answers_in_options = False

        q_result["valid"] = (
            q_result["checks"]["is_dict"]
            and q_result["checks"]["question_non_empty"]
            and q_result["checks"]["options_is_list"]
            and q_result["checks"]["options_length_valid"]
            and q_result["checks"]["options_non_empty"]
            and q_result["checks"]["options_unique"]
            and q_result["checks"]["answer_non_empty"]
            and q_result["checks"]["answer_in_options"]
        )

        result["question_checks"].append(q_result)

    if len(normalized_question_texts) != len(set(normalized_question_texts)):
        result["fatal_validation_errors"].append("duplicate_question_text")

    result["checks"]["all_questions_have_required_fields"] = all_required_fields
    result["checks"]["all_questions_non_empty"] = all_questions_non_empty
    result["checks"]["all_questions_unique"] = len(normalized_question_texts) == len(set(normalized_question_texts))
    result["checks"]["all_questions_have_3_options"] = all_questions_have_3_options
    result["checks"]["all_options_non_empty"] = all_options_non_empty
    result["checks"]["all_options_unique"] = all_options_unique
    result["checks"]["all_answers_non_empty"] = all_answers_non_empty
    result["checks"]["all_answers_in_options"] = all_answers_in_options

    result["validation_errors"] = (
        result["fatal_validation_errors"] + result["non_fatal_validation_errors"]
    )

    result["tolerant_valid"] = len(result["fatal_validation_errors"]) == 0
    result["recoverable"] = result["tolerant_valid"] and len(result["non_fatal_validation_errors"]) > 0

    return result