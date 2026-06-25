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

def _infer_match_top_level_fields_tolerant(response: dict) -> tuple[dict, list[str]]:
    """
    Tolerant inference for vocabulary matching top-level fields:
    expected fields:
    - exercise_type -> string
    - topic -> string
    - words -> list[str]
    - translations -> list[str]
    - answer_key -> dict[str, str]
    """
    errors: list[str] = []

    extracted = {
        "exercise_type": None,
        "topic": None,
        "words": None,
        "translations": None,
        "answer_key": None,
    }

    # exercise_type
    exercise_type, ex_status = _infer_field_by_position(
        response,
        ["exercise_type", "topic", "words", "translations", "answer_key"],
        "exercise_type",
    )
    if ex_status == "missing":
        errors.append("missing_exercise_type")
    else:
        if ex_status.startswith("inferred_by_position:"):
            errors.append(f"wrong_field_name_for_exercise_type:{ex_status.split(':', 1)[1]}")
        extracted["exercise_type"] = exercise_type

    # topic
    topic, topic_status = _infer_field_by_position(
        response,
        ["exercise_type", "topic", "words", "translations", "answer_key"],
        "topic",
    )
    if topic_status == "missing":
        errors.append("missing_topic")
    else:
        if topic_status.startswith("inferred_by_position:"):
            errors.append(f"wrong_field_name_for_topic:{topic_status.split(':', 1)[1]}")
        extracted["topic"] = topic

    # words
    if "words" in response:
        extracted["words"] = response["words"]
    else:
        list_candidates = [
            (k, v) for k, v in response.items()
            if isinstance(v, list) and all(isinstance(x, str) for x in v)
        ]

        if len(list_candidates) == 1:
            key, value = list_candidates[0]
            extracted["words"] = value
            errors.append(f"wrong_field_name_for_words:{key}")
        elif len(list_candidates) == 0:
            errors.append("missing_words")
        else:
            # spróbuj rozpoznać po pozycji
            words_by_pos, words_status = _infer_field_by_position(
                response,
                ["exercise_type", "topic", "words", "translations", "answer_key"],
                "words",
            )
            if words_status == "missing":
                errors.append("missing_words")
            elif isinstance(words_by_pos, list):
                extracted["words"] = words_by_pos
                if words_status.startswith("inferred_by_position:"):
                    errors.append(f"wrong_field_name_for_words:{words_status.split(':', 1)[1]}")
            else:
                errors.append("ambiguous_words_field")

    # translations
    if "translations" in response:
        extracted["translations"] = response["translations"]
    else:
        remaining_list_candidates = [
            (k, v) for k, v in response.items()
            if isinstance(v, list)
            and all(isinstance(x, str) for x in v)
            and v is not extracted["words"]
        ]

        if len(remaining_list_candidates) == 1:
            key, value = remaining_list_candidates[0]
            extracted["translations"] = value
            errors.append(f"wrong_field_name_for_translations:{key}")
        elif len(remaining_list_candidates) == 0:
            errors.append("missing_translations")
        else:
            translations_by_pos, translations_status = _infer_field_by_position(
                response,
                ["exercise_type", "topic", "words", "translations", "answer_key"],
                "translations",
            )
            if translations_status == "missing":
                errors.append("missing_translations")
            elif isinstance(translations_by_pos, list):
                extracted["translations"] = translations_by_pos
                if translations_status.startswith("inferred_by_position:"):
                    errors.append(
                        f"wrong_field_name_for_translations:{translations_status.split(':', 1)[1]}"
                    )
            else:
                errors.append("ambiguous_translations_field")

    # answer_key
    if "answer_key" in response:
        extracted["answer_key"] = response["answer_key"]
    else:
        dict_candidates = [
            (k, v) for k, v in response.items()
            if isinstance(v, dict)
        ]

        if len(dict_candidates) == 1:
            key, value = dict_candidates[0]
            extracted["answer_key"] = value
            errors.append(f"wrong_field_name_for_answer_key:{key}")
        elif len(dict_candidates) == 0:
            errors.append("missing_answer_key")
        else:
            answer_key_by_pos, answer_key_status = _infer_field_by_position(
                response,
                ["exercise_type", "topic", "words", "translations", "answer_key"],
                "answer_key",
            )
            if answer_key_status == "missing":
                errors.append("missing_answer_key")
            elif isinstance(answer_key_by_pos, dict):
                extracted["answer_key"] = answer_key_by_pos
                if answer_key_status.startswith("inferred_by_position:"):
                    errors.append(
                        f"wrong_field_name_for_answer_key:{answer_key_status.split(':', 1)[1]}"
                    )
            else:
                errors.append("missing_answer_key")

    return extracted, errors


def _normalize_str_list(values: list[str]) -> list[str]:
    return [_normalize_text_for_comparison(v) for v in values if isinstance(v, str)]


def _normalize_dict_str_str(data: dict) -> dict:
    normalized = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            normalized[_normalize_text_for_comparison(k)] = _normalize_text_for_comparison(v)
    return normalized


def _infer_vocabulary_definition_question_fields_tolerant(
    question_obj: dict,
    idx: int,
    validation_errors: list[str],
) -> dict:
    """
    Tolerant inference for vocabulary definition question:
    expected logical fields:
    - definition -> non-empty string
    - options -> list[str]
    - answer -> string
    """
    extracted = {
        "definition": None,
        "options": None,
        "answer": None,
    }

    used_keys = set()

    # exact canonical fields first
    if "definition" in question_obj:
        extracted["definition"] = question_obj["definition"]
        used_keys.add("definition")

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

    # infer definition
    if extracted["definition"] is None:
        if remaining_string_candidates:
            definition_key, definition_value = max(
                remaining_string_candidates,
                key=lambda item: len(item[1]),
            )
            extracted["definition"] = definition_value
            used_keys.add(definition_key)
            validation_errors.append(
                f"question_{idx}_wrong_field_name_for_definition:{definition_key}"
            )
        else:
            validation_errors.append(f"question_{idx}_missing_definition")

    # recompute remaining strings after taking definition
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
                    validation_errors.append(
                        f"question_{idx}_wrong_field_name_for_answer:{answer_key}"
                    )
                elif len(matching_candidates) == 0:
                    validation_errors.append(f"question_{idx}_missing_answer")
                else:
                    validation_errors.append(f"question_{idx}_ambiguous_answer_field")
            else:
                validation_errors.append(f"question_{idx}_ambiguous_answer_field")

    return extracted


def _validate_vocabulary_definition_question_tolerant(
    question_obj: Any,
    idx: int,
    options_count: int,
) -> tuple[list[str], dict, dict]:
    errors: list[str] = []

    checks = {
        "is_dict": False,
        "definition_non_empty": False,
        "options_is_list": False,
        "options_length_valid": False,
        "options_non_empty": False,
        "options_unique": False,
        "answer_non_empty": False,
        "answer_in_options": False,
        "definition_not_in_options": False,
    }

    if not isinstance(question_obj, dict):
        errors.append(f"question_{idx}_is_not_dict")
        return errors, checks, {
            "definition": None,
            "options": None,
            "answer": None,
        }

    checks["is_dict"] = True

    infer_errors: list[str] = []
    extracted = _infer_vocabulary_definition_question_fields_tolerant(
        question_obj,
        idx,
        infer_errors,
    )
    errors.extend(infer_errors)

    definition = extracted.get("definition")
    options = extracted.get("options")
    answer = extracted.get("answer")

    # definition
    if definition is None:
        pass
    elif _is_non_empty_string(definition):
        checks["definition_non_empty"] = True
    else:
        errors.append(f"question_{idx}_empty_definition")

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

    # definition should not appear in options
    if _is_non_empty_string(definition) and isinstance(options, list):
        normalized_definition = _normalize_text_for_comparison(definition)
        if normalized_options is None:
            normalized_options = [
                _normalize_text_for_comparison(opt)
                for opt in options
                if isinstance(opt, str)
            ]

        if normalized_definition in normalized_options:
            errors.append(f"question_{idx}_definition_present_in_options")
        else:
            checks["definition_not_in_options"] = True

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

            # synonym / antonym / definition
            "questions_is_list": False,
            "question_count_valid": False,
            "all_words_unique": None,
            "all_definitions_unique": None,
            "all_questions_tolerant_valid": False,

            # match
            "topic_valid": False,
            "words_is_list": False,
            "word_count_valid": None,
            "all_words_non_empty": None,
            "translations_is_list": False,
            "translation_count_valid": None,
            "all_translations_non_empty": None,
            "all_translations_unique": None,
            "answer_key_is_dict": False,
            "answer_key_count_valid": None,
            "answer_key_words_match_words": None,
            "answer_key_translations_in_translations": None,
            "answer_key_translations_unique": None,
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

    if mode_id == "vocabulary_match":
        extracted, infer_errors = _infer_match_top_level_fields_tolerant(response)

        for err in infer_errors:
            if _is_non_fatal_tolerant_error(err):
                result["non_fatal_validation_errors"].append(err)
            else:
                result["fatal_validation_errors"].append(err)

        exercise_type = extracted.get("exercise_type")
        topic = extracted.get("topic")
        words = extracted.get("words")
        translations = extracted.get("translations")
        answer_key = extracted.get("answer_key")

        # exercise_type
        if exercise_type == spec["exercise_type"]:
            result["checks"]["exercise_type_valid"] = True
        elif exercise_type is not None:
            result["fatal_validation_errors"].append("invalid_exercise_type")

        # topic
        result["checks"]["topic_valid"] = False
        if topic == spec["topic"]:
            result["checks"]["topic_valid"] = True
        elif topic is not None:
            result["fatal_validation_errors"].append("invalid_topic")

        # words
        result["checks"]["words_is_list"] = False
        result["checks"]["word_count_valid"] = None
        result["checks"]["all_words_non_empty"] = None
        result["checks"]["all_words_unique"] = None

        normalized_words = None

        if words is None:
            pass
        elif isinstance(words, list):
            result["checks"]["words_is_list"] = True

            if len(words) == spec["word_count"]:
                result["checks"]["word_count_valid"] = True
            else:
                result["checks"]["word_count_valid"] = False
                result["fatal_validation_errors"].append("invalid_word_count")

            if all(_is_non_empty_string(x) for x in words):
                result["checks"]["all_words_non_empty"] = True
                normalized_words = _normalize_str_list(words)
            else:
                result["checks"]["all_words_non_empty"] = False
                result["fatal_validation_errors"].append("empty_word")

            if normalized_words is not None and len(normalized_words) == len(set(normalized_words)):
                result["checks"]["all_words_unique"] = True
            else:
                result["checks"]["all_words_unique"] = False
                result["fatal_validation_errors"].append("duplicate_words")
        else:
            result["fatal_validation_errors"].append("words_is_not_list")

        # translations
        result["checks"]["translations_is_list"] = False
        result["checks"]["translation_count_valid"] = None
        result["checks"]["all_translations_non_empty"] = None
        result["checks"]["all_translations_unique"] = None

        normalized_translations = None

        if translations is None:
            pass
        elif isinstance(translations, list):
            result["checks"]["translations_is_list"] = True

            if len(translations) == spec["translation_count"]:
                result["checks"]["translation_count_valid"] = True
            else:
                result["checks"]["translation_count_valid"] = False
                result["fatal_validation_errors"].append("invalid_translation_count")

            if all(_is_non_empty_string(x) for x in translations):
                result["checks"]["all_translations_non_empty"] = True
                normalized_translations = _normalize_str_list(translations)
            else:
                result["checks"]["all_translations_non_empty"] = False
                result["fatal_validation_errors"].append("empty_translation")

            if normalized_translations is not None and len(normalized_translations) == len(set(normalized_translations)):
                result["checks"]["all_translations_unique"] = True
            else:
                result["checks"]["all_translations_unique"] = False
                result["fatal_validation_errors"].append("duplicate_translations")
        else:
            result["fatal_validation_errors"].append("translations_is_not_list")

        # answer_key
        result["checks"]["answer_key_is_dict"] = False
        result["checks"]["answer_key_count_valid"] = None
        result["checks"]["answer_key_words_match_words"] = None
        result["checks"]["answer_key_translations_in_translations"] = None
        result["checks"]["answer_key_translations_unique"] = None

        if answer_key is None:
            pass
        elif isinstance(answer_key, dict):
            result["checks"]["answer_key_is_dict"] = True

            if len(answer_key) == spec["word_count"]:
                result["checks"]["answer_key_count_valid"] = True
            else:
                result["checks"]["answer_key_count_valid"] = False
                result["fatal_validation_errors"].append("invalid_answer_key_count")

            empty_key_found = False
            empty_value_found = False

            for k, v in answer_key.items():
                if not _is_non_empty_string(k):
                    empty_key_found = True
                if not _is_non_empty_string(v):
                    empty_value_found = True

            if empty_key_found:
                result["fatal_validation_errors"].append("empty_answer_key_word")
            if empty_value_found:
                result["fatal_validation_errors"].append("empty_answer_key_translation")

            normalized_answer_key = _normalize_dict_str_str(answer_key)

            # klucze muszą odpowiadać words
            if normalized_words is not None:
                words_set = set(normalized_words)
                answer_key_words_set = set(normalized_answer_key.keys())

                missing_words = words_set - answer_key_words_set
                unknown_words = answer_key_words_set - words_set

                if not missing_words and not unknown_words:
                    result["checks"]["answer_key_words_match_words"] = True
                else:
                    result["checks"]["answer_key_words_match_words"] = False
                    for _ in missing_words:
                        result["fatal_validation_errors"].append("answer_key_missing_word")
                    for _ in unknown_words:
                        result["fatal_validation_errors"].append("answer_key_unknown_word")

            # wartości muszą należeć do translations
            if normalized_translations is not None:
                translations_set = set(normalized_translations)
                bad_translation_values = [
                    value for value in normalized_answer_key.values()
                    if value not in translations_set
                ]

                if not bad_translation_values:
                    result["checks"]["answer_key_translations_in_translations"] = True
                else:
                    result["checks"]["answer_key_translations_in_translations"] = False
                    for _ in bad_translation_values:
                        result["fatal_validation_errors"].append(
                            "answer_key_translation_not_in_translations"
                        )

            # wartości w answer_key powinny być unikalne
            normalized_answer_values = list(normalized_answer_key.values())
            if len(normalized_answer_values) == len(set(normalized_answer_values)):
                result["checks"]["answer_key_translations_unique"] = True
            else:
                result["checks"]["answer_key_translations_unique"] = False
                result["fatal_validation_errors"].append("duplicate_answer_key_translations")

        else:
            result["fatal_validation_errors"].append("answer_key_is_not_dict")

        return _finalize_tolerant_result(result)
    
    if mode_id == "vocabulary_definition":
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
        normalized_definitions = []

        for idx, q in enumerate(questions, start=1):
            q_errors, q_checks, extracted = _validate_vocabulary_definition_question_tolerant(
                q,
                idx,
                options_count=spec["options_count"],
            )

            for err in q_errors:
                if _is_non_fatal_tolerant_error(err):
                    result["non_fatal_validation_errors"].append(err)
                else:
                    result["fatal_validation_errors"].append(err)

            definition = extracted.get("definition")
            if _is_non_empty_string(definition):
                normalized_definitions.append(_normalize_text_for_comparison(definition))

            result["question_checks"].append({
                "question_index": idx,
                "checks": q_checks,
                "valid": len([e for e in q_errors if not _is_non_fatal_tolerant_error(e)]) == 0,
                "extracted_fields": extracted,
            })

            if any(not _is_non_fatal_tolerant_error(err) for err in q_errors):
                all_questions_tolerant_valid = False

        result["checks"]["all_questions_tolerant_valid"] = all_questions_tolerant_valid

        if len(normalized_definitions) == len(set(normalized_definitions)):
            result["checks"]["all_definitions_unique"] = True
        else:
            result["checks"]["all_definitions_unique"] = False
            result["fatal_validation_errors"].append("duplicate_definition")

        return _finalize_tolerant_result(result)
    
    result["fatal_validation_errors"].append("unsupported_vocabulary_mode")
    return _finalize_tolerant_result(result)