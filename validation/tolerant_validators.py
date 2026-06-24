from typing import Any

from validation.specs import (
    TEST_SECTION_SPECS,
    GRAMMAR_MODE_SPECS,
    WRITING_MODE_SPECS,
    VOCABULARY_MODE_SPECS,
)

def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finalize_tolerant_result(result: dict) -> dict:
    result["validation_errors"] = (
        result["fatal_validation_errors"] + result["non_fatal_validation_errors"]
    )
    result["tolerant_valid"] = len(result["fatal_validation_errors"]) == 0
    result["recoverable"] = (
        result["tolerant_valid"] and len(result["non_fatal_validation_errors"]) > 0
    )
    return result


def _is_non_fatal_tolerant_error(error: str) -> bool:
    return (
        error.startswith("wrong_field_name_for_")
        or "_wrong_field_name_for_" in error
    )


def _infer_list_field_by_role(
    obj: dict,
    canonical_name: str,
):
    """
    Returns:
    - value
    - status: exact / inferred:<actual_key> / missing / ambiguous
    """
    if canonical_name in obj:
        return obj[canonical_name], "exact"

    list_candidates = [
        (k, v) for k, v in obj.items()
        if isinstance(v, list)
    ]

    if len(list_candidates) == 1:
        key, value = list_candidates[0]
        return value, f"inferred:{key}"

    if len(list_candidates) == 0:
        return None, "missing"

    return None, "ambiguous"


def _infer_field_by_position(
    obj: dict,
    expected_order: list[str],
    canonical_name: str,
):
    """
    Returns:
    - value
    - status: exact / inferred_by_position:<actual_key> / missing
    """
    if canonical_name in obj:
        return obj[canonical_name], "exact"

    items = list(obj.items())

    try:
        idx = expected_order.index(canonical_name)
    except ValueError:
        return None, "missing"

    if idx < len(items):
        actual_key, actual_value = items[idx]
        return actual_value, f"inferred_by_position:{actual_key}"

    return None, "missing"


def _infer_question_fields(question_obj: dict, idx: int, validation_errors: list[str]) -> dict:
    extracted = {
        "question_number": None,
        "question": None,
        "options": None,
        "answer": None
    }

    used_keys = set()

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

    remaining_string_candidates = [
        (k, v) for k, v in question_obj.items()
        if isinstance(v, str) and k not in used_keys
    ]

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

    if question_text is None:
        pass
    elif _is_non_empty_string(question_text):
        checks["question_non_empty"] = True
    else:
        errors.append(f"question_{idx}_empty_question")

    if options is None:
        pass
    elif isinstance(options, list):
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


def _infer_grammar_question_fields_tolerant(question_obj: dict, idx: int, validation_errors: list[str]) -> dict:
    extracted = {
        "question": None,
        "options": None,
        "answer": None,
    }

    used_keys = set()

    if "question" in question_obj:
        extracted["question"] = question_obj["question"]
        used_keys.add("question")

    if "options" in question_obj:
        extracted["options"] = question_obj["options"]
        used_keys.add("options")

    if "answer" in question_obj:
        extracted["answer"] = question_obj["answer"]
        used_keys.add("answer")

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

    remaining_string_candidates = [
        (k, v) for k, v in question_obj.items()
        if isinstance(v, str) and k not in used_keys
    ]

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

def _infer_vocabulary_choice_question_fields_tolerant(
    question_obj: dict,
    idx: int,
    validation_errors: list[str],
) -> dict:
    """
    Tolerant inference for vocabulary synonym/antonym question:
    expected logical fields:
    - word -> non-empty string
    - options -> list[str]
    - answer -> string
    """
    extracted = {
        "word": None,
        "options": None,
        "answer": None,
    }

    used_keys = set()

    # exact canonical fields first
    if "word" in question_obj:
        extracted["word"] = question_obj["word"]
        used_keys.add("word")

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

    # remaining strings
    remaining_string_candidates = [
        (k, v) for k, v in question_obj.items()
        if isinstance(v, str) and k not in used_keys
    ]

    # infer word
    if extracted["word"] is None:
        if remaining_string_candidates:
            word_key, word_value = max(remaining_string_candidates, key=lambda item: len(item[1]))
            extracted["word"] = word_value
            used_keys.add(word_key)
            validation_errors.append(f"question_{idx}_wrong_field_name_for_word:{word_key}")
        else:
            validation_errors.append(f"question_{idx}_missing_word")

    # recompute remaining strings after taking word
    remaining_string_candidates = [
        (k, v) for k, v in question_obj.items()
        if isinstance(v, str) and k not in used_keys
    ]

    # infer answer
    if extracted["answer"] is None:
        if len(remaining_string_candidates) == 1:
            answer_key, answer_value = remaining_string_candidates[0]
            extracted["answer"] = answer_value
            used_keys.add(answer_key)
            validation_errors.append(f"question_{idx}_wrong_field_name_for_answer:{answer_key}")
        elif len(remaining_string_candidates) == 0:
            validation_errors.append(f"question_{idx}_missing_answer")
        else:
            if extracted["options"] is not None:
                matching_candidates = [
                    (k, v) for k, v in remaining_string_candidates
                    if v in extracted["options"]
                ]
                if len(matching_candidates) == 1:
                    answer_key, answer_value = matching_candidates[0]
                    extracted["answer"] = answer_value
                    used_keys.add(answer_key)
                    validation_errors.append(f"question_{idx}_wrong_field_name_for_answer:{answer_key}")
                elif len(matching_candidates) == 0:
                    validation_errors.append(f"question_{idx}_missing_answer")
                else:
                    validation_errors.append(f"question_{idx}_ambiguous_answer_field")
            else:
                validation_errors.append(f"question_{idx}_ambiguous_answer_field")

    return extracted

def _normalize_text_for_comparison(value: str) -> str:
    return value.strip().lower()


def _validate_vocabulary_choice_question_tolerant(
    question_obj: Any,
    idx: int,
    options_count: int,
) -> tuple[list[str], dict, dict]:
    errors: list[str] = []

    checks = {
        "is_dict": False,
        "word_non_empty": False,
        "options_is_list": False,
        "options_length_valid": False,
        "options_non_empty": False,
        "options_unique": False,
        "answer_non_empty": False,
        "answer_in_options": False,
        "word_not_in_options": False,
    }

    if not isinstance(question_obj, dict):
        errors.append(f"question_{idx}_is_not_dict")
        return errors, checks, {
            "word": None,
            "options": None,
            "answer": None,
        }

    checks["is_dict"] = True

    infer_errors: list[str] = []
    extracted = _infer_vocabulary_choice_question_fields_tolerant(question_obj, idx, infer_errors)
    errors.extend(infer_errors)

    word = extracted.get("word")
    options = extracted.get("options")
    answer = extracted.get("answer")

    # word
    if word is None:
        pass
    elif _is_non_empty_string(word):
        checks["word_non_empty"] = True
    else:
        errors.append(f"question_{idx}_empty_word")

    # options
    normalized_options = None

    if options is None:
        pass
    elif isinstance(options, list):
        checks["options_is_list"] = True

        if len(options) == options_count:
            checks["options_length_valid"] = True
        else:
            errors.append(f"question_{idx}_options_not_length_{options_count}")

        if all(_is_non_empty_string(opt) for opt in options):
            checks["options_non_empty"] = True
            normalized_options = [_normalize_text_for_comparison(opt) for opt in options]
        else:
            errors.append(f"question_{idx}_empty_option")

        if normalized_options is not None and len(normalized_options) == len(set(normalized_options)):
            checks["options_unique"] = True
        else:
            errors.append(f"question_{idx}_duplicate_options")
    else:
        errors.append(f"question_{idx}_options_is_not_list")

    # answer
    if answer is None:
        pass
    elif _is_non_empty_string(answer):
        checks["answer_non_empty"] = True
    else:
        errors.append(f"question_{idx}_empty_answer")

    if isinstance(options, list) and _is_non_empty_string(answer):
        normalized_answer = _normalize_text_for_comparison(answer)
        if normalized_options is None:
            normalized_options = [
                _normalize_text_for_comparison(opt)
                for opt in options
                if isinstance(opt, str)
            ]

        if normalized_answer in normalized_options:
            checks["answer_in_options"] = True
        else:
            errors.append(f"question_{idx}_answer_not_in_options")

    # word should not appear in options
    if _is_non_empty_string(word) and isinstance(options, list):
        normalized_word = _normalize_text_for_comparison(word)
        if normalized_options is None:
            normalized_options = [
                _normalize_text_for_comparison(opt)
                for opt in options
                if isinstance(opt, str)
            ]

        if normalized_word in normalized_options:
            errors.append(f"question_{idx}_word_present_in_options")
        else:
            checks["word_not_in_options"] = True

    return errors, checks, extracted

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
        result["fatal_validation_errors"].append("unknown_mode_id")
        return _finalize_tolerant_result(result)

    if not attempt_data.get("success"):
        result["fatal_validation_errors"].append("original_attempt_not_successful")
        return _finalize_tolerant_result(result)

    response = attempt_data.get("response")
    if not isinstance(response, dict):
        result["fatal_validation_errors"].append("response_is_not_dict")
        return _finalize_tolerant_result(result)

    result["checks"]["response_is_dict"] = True

    expected_top_order = [
        "exercise_type",
        "section_id",
        "section_title",
        "cefr_range",
    ]

    exercise_type, ex_status = _infer_field_by_position(response, expected_top_order, "exercise_type")
    if ex_status == "missing":
        result["fatal_validation_errors"].append("missing_exercise_type")
    else:
        result["checks"]["exercise_type_present_or_inferred"] = True
        if ex_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_exercise_type:{ex_status.split(':', 1)[1]}"
            )
        if exercise_type == "CEFR English Placement Test":
            result["checks"]["exercise_type_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_exercise_type")

    section_id, sid_status = _infer_field_by_position(response, expected_top_order, "section_id")
    if sid_status == "missing":
        result["fatal_validation_errors"].append("missing_section_id")
    else:
        result["checks"]["section_id_present_or_inferred"] = True
        if sid_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_section_id:{sid_status.split(':', 1)[1]}"
            )
        if section_id == spec["section_id"]:
            result["checks"]["section_id_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_section_id")

    section_title, st_status = _infer_field_by_position(response, expected_top_order, "section_title")
    if st_status == "missing":
        result["fatal_validation_errors"].append("missing_section_title")
    else:
        result["checks"]["section_title_present_or_inferred"] = True
        if st_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_section_title:{st_status.split(':', 1)[1]}"
            )
        if section_title == spec["section_title"]:
            result["checks"]["section_title_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_section_title")

    cefr_range, cr_status = _infer_field_by_position(response, expected_top_order, "cefr_range")
    if cr_status == "missing":
        result["fatal_validation_errors"].append("missing_cefr_range")
    else:
        result["checks"]["cefr_range_present_or_inferred"] = True
        if cr_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_cefr_range:{cr_status.split(':', 1)[1]}"
            )
        if cefr_range == spec["cefr_range"]:
            result["checks"]["cefr_range_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_cefr_range")

    questions, q_status = _infer_list_field_by_role(response, "questions")
    if q_status == "missing":
        result["fatal_validation_errors"].append("missing_questions")
        return _finalize_tolerant_result(result)
    elif q_status == "ambiguous":
        result["fatal_validation_errors"].append("ambiguous_questions_field")
        return _finalize_tolerant_result(result)
    else:
        if q_status.startswith("inferred:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_questions:{q_status.split(':', 1)[1]}"
            )

    if not isinstance(questions, list):
        result["fatal_validation_errors"].append("questions_is_not_list")
        return _finalize_tolerant_result(result)

    result["checks"]["questions_is_list"] = True

    if len(questions) == spec["question_count"]:
        result["checks"]["question_count_valid"] = True
    else:
        result["fatal_validation_errors"].append("invalid_question_count")

    actual_numbers: list[int] = []
    normalized_question_texts = []
    numbering_assessable = True
    all_questions_tolerant_valid = True

    for idx, q in enumerate(questions, start=1):
        q_errors, q_checks, extracted = _validate_question_tolerant(q, idx)

        for err in q_errors:
            if _is_non_fatal_tolerant_error(err):
                result["non_fatal_validation_errors"].append(err)
            else:
                result["fatal_validation_errors"].append(err)

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
            result["fatal_validation_errors"].append("invalid_question_numbering")
    else:
        result["checks"]["question_numbering_valid"] = None

    if len(normalized_question_texts) != len(set(normalized_question_texts)):
        result["fatal_validation_errors"].append("duplicate_question_text")
        result["checks"]["all_questions_tolerant_valid"] = False

    requires_reading_text = spec["requires_reading_text"]

    if requires_reading_text:
        reading_text, rt_status = _infer_field_by_position(response, ["reading_text"], "reading_text")
        if rt_status == "missing":
            result["checks"]["reading_text_valid"] = False
            result["fatal_validation_errors"].append("missing_reading_text")
        else:
            if rt_status.startswith("inferred_by_position:"):
                result["non_fatal_validation_errors"].append(
                    f"wrong_field_name_for_reading_text:{rt_status.split(':', 1)[1]}"
                )
            if _is_non_empty_string(reading_text):
                result["checks"]["reading_text_valid"] = True
            else:
                result["checks"]["reading_text_valid"] = False
                result["fatal_validation_errors"].append("empty_reading_text")
    else:
        if "reading_text" in response and response.get("reading_text"):
            result["checks"]["reading_text_valid"] = False
            result["fatal_validation_errors"].append("unexpected_reading_text")
        else:
            result["checks"]["reading_text_valid"] = True

    return _finalize_tolerant_result(result)


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
        return _finalize_tolerant_result(result)

    if not attempt_data.get("success"):
        result["fatal_validation_errors"].append("original_attempt_not_successful")
        return _finalize_tolerant_result(result)

    response = attempt_data.get("response")
    if not isinstance(response, dict):
        result["fatal_validation_errors"].append("response_is_not_dict")
        return _finalize_tolerant_result(result)

    result["checks"]["response_is_dict"] = True

    expected_top_order = [
        "exercise_type",
        "grammar_topic",
        "level",
    ]

    exercise_type, ex_status = _infer_field_by_position(response, expected_top_order, "exercise_type")
    if ex_status == "missing":
        result["fatal_validation_errors"].append("missing_exercise_type")
    else:
        if ex_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_exercise_type:{ex_status.split(':', 1)[1]}"
            )
        if exercise_type == spec["exercise_type"]:
            result["checks"]["exercise_type_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_exercise_type")

    grammar_topic, gt_status = _infer_field_by_position(response, expected_top_order, "grammar_topic")
    if gt_status == "missing":
        result["fatal_validation_errors"].append("missing_grammar_topic")
    else:
        if gt_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_grammar_topic:{gt_status.split(':', 1)[1]}"
            )
        if grammar_topic == spec["grammar_topic"]:
            result["checks"]["grammar_topic_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_grammar_topic")

    level, lv_status = _infer_field_by_position(response, expected_top_order, "level")
    if lv_status == "missing":
        result["fatal_validation_errors"].append("missing_level")
    else:
        if lv_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_level:{lv_status.split(':', 1)[1]}"
            )
        if level == spec["level"]:
            result["checks"]["level_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_level")

    questions, q_status = _infer_list_field_by_role(response, "questions")
    if q_status == "missing":
        result["fatal_validation_errors"].append("missing_questions")
        return _finalize_tolerant_result(result)
    elif q_status == "ambiguous":
        result["fatal_validation_errors"].append("ambiguous_questions_field")
        return _finalize_tolerant_result(result)
    else:
        if q_status.startswith("inferred:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_questions:{q_status.split(':', 1)[1]}"
            )

    if not isinstance(questions, list):
        result["fatal_validation_errors"].append("questions_is_not_list")
        return _finalize_tolerant_result(result)

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

        infer_errors: list[str] = []
        extracted = _infer_grammar_question_fields_tolerant(q, idx, infer_errors)

        for err in infer_errors:
            if _is_non_fatal_tolerant_error(err):
                result["non_fatal_validation_errors"].append(err)
            else:
                result["fatal_validation_errors"].append(err)

        question_text = extracted.get("question")
        options = extracted.get("options")
        answer = extracted.get("answer")

        if question_text is None or options is None or answer is None:
            all_required_fields = False

        if question_text is None:
            all_questions_non_empty = False
        elif _is_non_empty_string(question_text):
            q_result["checks"]["question_non_empty"] = True
            normalized_question_texts.append(question_text.strip().lower())
        else:
            result["fatal_validation_errors"].append(f"question_{idx}_empty_question")
            all_questions_non_empty = False

        if options is None:
            all_questions_have_3_options = False
            all_options_non_empty = False
            all_options_unique = False
        elif isinstance(options, list):
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

        if answer is None:
            all_answers_non_empty = False
            all_answers_in_options = False
        elif _is_non_empty_string(answer):
            q_result["checks"]["answer_non_empty"] = True

            if isinstance(options, list):
                if answer in options:
                    q_result["checks"]["answer_in_options"] = True
                else:
                    result["fatal_validation_errors"].append(f"question_{idx}_answer_not_in_options")
                    all_answers_in_options = False
            else:
                all_answers_in_options = False
        else:
            result["fatal_validation_errors"].append(f"question_{idx}_empty_answer")
            all_answers_non_empty = False
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

    return _finalize_tolerant_result(result)


def validate_writing_tolerant(attempt_data: dict) -> dict:
    mode_id = attempt_data.get("mode_id")
    spec = WRITING_MODE_SPECS.get(mode_id)

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
            "learner_native_language_valid": False,
            "task_content_valid": False,
            "original_text_valid": False,
            "corrected_text_non_empty": False,
            "learner_errors_is_list": False,
            "content_compliance_valid": False,
            "content_feedback_non_empty": False,
            "all_learner_errors_tolerant_valid": True,
        },
        "learner_error_checks": []
    }

    if spec is None:
        result["fatal_validation_errors"].append("unknown_mode_id")
        return _finalize_tolerant_result(result)

    if not attempt_data.get("success"):
        result["fatal_validation_errors"].append("original_attempt_not_successful")
        return _finalize_tolerant_result(result)

    response = attempt_data.get("response")
    if not isinstance(response, dict):
        result["fatal_validation_errors"].append("response_is_not_dict")
        return _finalize_tolerant_result(result)

    result["checks"]["response_is_dict"] = True

    expected_top_order = [
        "exercise_type",
        "learner_native_language",
        "task_content",
        "original_text",
        "corrected_text",
        "learner_errors",
        "content_compliance",
        "content_feedback",
    ]

    exercise_type, ex_status = _infer_field_by_position(response, expected_top_order, "exercise_type")
    if ex_status == "missing":
        result["fatal_validation_errors"].append("missing_exercise_type")
    else:
        if ex_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_exercise_type:{ex_status.split(':', 1)[1]}"
            )
        if exercise_type == spec["exercise_type"]:
            result["checks"]["exercise_type_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_exercise_type")

    learner_native_language, ln_status = _infer_field_by_position(
        response, expected_top_order, "learner_native_language"
    )
    if ln_status == "missing":
        result["fatal_validation_errors"].append("missing_learner_native_language")
    else:
        if ln_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_learner_native_language:{ln_status.split(':', 1)[1]}"
            )
        if learner_native_language == spec["learner_native_language"]:
            result["checks"]["learner_native_language_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_learner_native_language")

    task_content, tc_status = _infer_field_by_position(response, expected_top_order, "task_content")
    if tc_status == "missing":
        result["fatal_validation_errors"].append("missing_task_content")
    else:
        if tc_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_task_content:{tc_status.split(':', 1)[1]}"
            )
        if task_content == spec["task_content"]:
            result["checks"]["task_content_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_task_content")

    original_text, ot_status = _infer_field_by_position(response, expected_top_order, "original_text")
    if ot_status == "missing":
        result["fatal_validation_errors"].append("missing_original_text")
    else:
        if ot_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_original_text:{ot_status.split(':', 1)[1]}"
            )
        if original_text == spec["original_text"]:
            result["checks"]["original_text_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_original_text")

    corrected_text, ct_status = _infer_field_by_position(response, expected_top_order, "corrected_text")
    if ct_status == "missing":
        result["fatal_validation_errors"].append("missing_corrected_text")
    else:
        if ct_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_corrected_text:{ct_status.split(':', 1)[1]}"
            )
        if _is_non_empty_string(corrected_text):
            result["checks"]["corrected_text_non_empty"] = True
        else:
            result["fatal_validation_errors"].append("empty_corrected_text")

    learner_errors, le_status = _infer_list_field_by_role(response, "learner_errors")
    if le_status == "missing":
        result["fatal_validation_errors"].append("missing_learner_errors")
        learner_errors = None
    elif le_status == "ambiguous":
        result["fatal_validation_errors"].append("ambiguous_learner_errors_field")
        learner_errors = None
    else:
        if le_status.startswith("inferred:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_learner_errors:{le_status.split(':', 1)[1]}"
            )
        if isinstance(learner_errors, list):
            result["checks"]["learner_errors_is_list"] = True
        else:
            result["fatal_validation_errors"].append("learner_errors_is_not_list")
            learner_errors = None

    content_compliance, cc_status = _infer_field_by_position(response, expected_top_order, "content_compliance")
    if cc_status == "missing":
        result["fatal_validation_errors"].append("missing_content_compliance")
    else:
        if cc_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_content_compliance:{cc_status.split(':', 1)[1]}"
            )
        if content_compliance in spec["allowed_content_compliance"]:
            result["checks"]["content_compliance_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_content_compliance")

    content_feedback, cf_status = _infer_field_by_position(response, expected_top_order, "content_feedback")
    if cf_status == "missing":
        result["fatal_validation_errors"].append("missing_content_feedback")
    else:
        if cf_status.startswith("inferred_by_position:"):
            result["non_fatal_validation_errors"].append(
                f"wrong_field_name_for_content_feedback:{cf_status.split(':', 1)[1]}"
            )
        if _is_non_empty_string(content_feedback):
            result["checks"]["content_feedback_non_empty"] = True
        else:
            result["fatal_validation_errors"].append("empty_content_feedback")

    all_learner_errors_tolerant_valid = True

    if isinstance(learner_errors, list):
        expected_error_order = ["error", "explanation"]

        for idx, item in enumerate(learner_errors, start=1):
            item_result = {
                "learner_error_index": idx,
                "checks": {
                    "is_dict": False,
                    "error_non_empty": False,
                    "explanation_non_empty": False,
                },
                "valid": False
            }

            if not isinstance(item, dict):
                result["fatal_validation_errors"].append(f"learner_error_{idx}_is_not_dict")
                result["learner_error_checks"].append(item_result)
                all_learner_errors_tolerant_valid = False
                continue

            item_result["checks"]["is_dict"] = True

            error_text, err_status = _infer_field_by_position(item, expected_error_order, "error")
            if err_status == "missing":
                result["fatal_validation_errors"].append(f"learner_error_{idx}_missing_error")
            else:
                if err_status.startswith("inferred_by_position:"):
                    result["non_fatal_validation_errors"].append(
                        f"learner_error_{idx}_wrong_field_name_for_error:{err_status.split(':', 1)[1]}"
                    )
                if _is_non_empty_string(error_text):
                    item_result["checks"]["error_non_empty"] = True
                else:
                    result["fatal_validation_errors"].append(f"learner_error_{idx}_empty_error")

            explanation, exp_status = _infer_field_by_position(item, expected_error_order, "explanation")
            if exp_status == "missing":
                result["fatal_validation_errors"].append(f"learner_error_{idx}_missing_explanation")
            else:
                if exp_status.startswith("inferred_by_position:"):
                    result["non_fatal_validation_errors"].append(
                        f"learner_error_{idx}_wrong_field_name_for_explanation:{exp_status.split(':', 1)[1]}"
                    )
                if _is_non_empty_string(explanation):
                    item_result["checks"]["explanation_non_empty"] = True
                else:
                    result["fatal_validation_errors"].append(f"learner_error_{idx}_empty_explanation")

            item_result["valid"] = all(item_result["checks"].values())
            if not item_result["valid"]:
                all_learner_errors_tolerant_valid = False

            result["learner_error_checks"].append(item_result)

    result["checks"]["all_learner_errors_tolerant_valid"] = all_learner_errors_tolerant_valid

    return _finalize_tolerant_result(result)

def validate_vocabulary_tolerant(attempt_data: dict) -> dict:
    mode_id = attempt_data.get("mode_id")
    spec = VOCABULARY_MODE_SPECS.get(mode_id)

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
            "questions_is_list": False,
            "question_count_valid": False,
            "all_words_unique": None,
            "all_questions_tolerant_valid": False,
        },
        "question_checks": []
    }

    if spec is None:
        result["fatal_validation_errors"].append("unknown_mode_id")
        return _finalize_tolerant_result(result)

    if not attempt_data.get("success"):
        result["fatal_validation_errors"].append("original_attempt_not_successful")
        return _finalize_tolerant_result(result)

    response = attempt_data.get("response")
    if not isinstance(response, dict):
        result["fatal_validation_errors"].append("response_is_not_dict")
        return _finalize_tolerant_result(result)

    result["checks"]["response_is_dict"] = True

    # synonym / antonym branch
    if mode_id in {"vocabulary_synonym", "vocabulary_antonym"}:
        expected_top_order = [
            "exercise_type",
            "questions",
        ]

        exercise_type, ex_status = _infer_field_by_position(response, expected_top_order, "exercise_type")
        if ex_status == "missing":
            result["fatal_validation_errors"].append("missing_exercise_type")
        else:
            if ex_status.startswith("inferred_by_position:"):
                result["non_fatal_validation_errors"].append(
                    f"wrong_field_name_for_exercise_type:{ex_status.split(':', 1)[1]}"
                )
            if exercise_type == spec["exercise_type"]:
                result["checks"]["exercise_type_valid"] = True
            else:
                result["fatal_validation_errors"].append("invalid_exercise_type")

        questions, q_status = _infer_list_field_by_role(response, "questions")

        if q_status == "missing":
            result["fatal_validation_errors"].append("missing_questions")
            return _finalize_tolerant_result(result)
        elif q_status == "ambiguous":
            result["fatal_validation_errors"].append("ambiguous_questions_field")
            return _finalize_tolerant_result(result)
        else:
            if q_status.startswith("inferred:"):
                result["non_fatal_validation_errors"].append(
                    f"wrong_field_name_for_questions:{q_status.split(':', 1)[1]}"
                )

        if not isinstance(questions, list):
            result["fatal_validation_errors"].append("questions_is_not_list")
            return _finalize_tolerant_result(result)

        result["checks"]["questions_is_list"] = True

        if len(questions) == spec["question_count"]:
            result["checks"]["question_count_valid"] = True
        else:
            result["fatal_validation_errors"].append("invalid_question_count")

        all_questions_tolerant_valid = True
        normalized_words = []

        for idx, q in enumerate(questions, start=1):
            q_errors, q_checks, extracted = _validate_vocabulary_choice_question_tolerant(
                q,
                idx,
                options_count=spec["options_count"],
            )

            for err in q_errors:
                if _is_non_fatal_tolerant_error(err):
                    result["non_fatal_validation_errors"].append(err)
                else:
                    result["fatal_validation_errors"].append(err)

            word = extracted.get("word")
            if _is_non_empty_string(word):
                normalized_words.append(_normalize_text_for_comparison(word))

            result["question_checks"].append({
                "question_index": idx,
                "checks": q_checks,
                "valid": len([e for e in q_errors if not _is_non_fatal_tolerant_error(e)]) == 0,
                "extracted_fields": extracted,
            })

            if any(not _is_non_fatal_tolerant_error(err) for err in q_errors):
                all_questions_tolerant_valid = False

        result["checks"]["all_questions_tolerant_valid"] = all_questions_tolerant_valid

        if len(normalized_words) == len(set(normalized_words)):
            result["checks"]["all_words_unique"] = True
        else:
            result["checks"]["all_words_unique"] = False
            result["fatal_validation_errors"].append("duplicate_word")

        return _finalize_tolerant_result(result)

    result["fatal_validation_errors"].append("unsupported_vocabulary_mode")
    return _finalize_tolerant_result(result)