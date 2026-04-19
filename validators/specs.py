TEST_SECTION_SPECS = {
    "test_section_1": {
        "family": "test",
        "schema_type": "test_standard_section",
        "section_id": 1,
        "section_title": "Grammar and Vocabulary (A1-A2)",
        "cefr_range": "A1-A2",
        "question_count": 10,
        "question_numbers": list(range(1, 11)),
        "requires_reading_text": False,
    },
    "test_section_2": {
        "family": "test",
        "schema_type": "test_standard_section",
        "section_id": 2,
        "section_title": "Grammar and Vocabulary (B1)",
        "cefr_range": "B1",
        "question_count": 10,
        "question_numbers": list(range(11, 21)),
        "requires_reading_text": False,
    },
    "test_section_3": {
        "family": "test",
        "schema_type": "test_standard_section",
        "section_id": 3,
        "section_title": "Grammar and Vocabulary (B2)",
        "cefr_range": "B2",
        "question_count": 10,
        "question_numbers": list(range(21, 31)),
        "requires_reading_text": False,
    },
    "test_section_4": {
        "family": "test",
        "schema_type": "test_standard_section",
        "section_id": 4,
        "section_title": "Advanced Grammar and Structures (C1-C2)",
        "cefr_range": "C1-C2",
        "question_count": 5,
        "question_numbers": list(range(31, 36)),
        "requires_reading_text": False,
    },
    "test_section_5": {
        "family": "test",
        "schema_type": "test_reading_section",
        "section_id": 5,
        "section_title": "Reading Comprehension",
        "cefr_range": "B2-C1",
        "question_count": 5,
        "question_numbers": list(range(36, 41)),
        "requires_reading_text": True,
    },
    "test_section_6": {
        "family": "test",
        "schema_type": "test_standard_section",
        "section_id": 6,
        "section_title": "Language Functions",
        "cefr_range": "B1-B2",
        "question_count": 5,
        "question_numbers": list(range(41, 46)),
        "requires_reading_text": False,
    },
    "test_section_7": {
        "family": "test",
        "schema_type": "test_standard_section",
        "section_id": 7,
        "section_title": "Advanced Vocabulary",
        "cefr_range": "B2-C1",
        "question_count": 5,
        "question_numbers": list(range(46, 51)),
        "requires_reading_text": False,
    },
}

GRAMMAR_MODE_SPECS = {
    "grammar_conditionals_1_2_3": {
        "family": "grammar",
        "schema_type": "grammar_multiple_choice",
        "exercise_type": "grammar_multiple_choice",
        "grammar_topic": "conditionals_1_2_3",
        "level": "B1-B2",
        "question_count": 10,
        "options_count": 3,
    },
    "grammar_gerund_vs_infinitive": {
        "family": "grammar",
        "schema_type": "grammar_multiple_choice",
        "exercise_type": "grammar_multiple_choice",
        "grammar_topic": "gerund_vs_infinitive",
        "level": "B1-B2",
        "question_count": 10,
        "options_count": 3,
    },
    "grammar_past_simple_vs_present_perfect": {
        "family": "grammar",
        "schema_type": "grammar_multiple_choice",
        "exercise_type": "grammar_multiple_choice",
        "grammar_topic": "past_simple_vs_present_perfect",
        "level": "A2-B1",
        "question_count": 10,
        "options_count": 3,
    },
    "grammar_simple_vs_continuous": {
        "family": "grammar",
        "schema_type": "grammar_multiple_choice",
        "exercise_type": "grammar_multiple_choice",
        "grammar_topic": "simple_vs_continuous",
        "level": "A2-B1",
        "question_count": 10,
        "options_count": 3,
    },
}

WRITING_MODE_SPECS = {
    "writing": {
        "family": "writing",
        "schema_type": "writing_feedback",
        "exercise_type": "Writing",
        "learner_native_language": "Polish",
        "task_content": "Write a letter to a friend from England.\nYou must include the following information:\n- how old your cousin is,\n- what you did at the birthday party,\n- what the cake tasted like.",
        "original_text": "Last weekend I was on my cousin birthday party.\nShe have 18 years old.\nWe dancing and play games all night.\nThe cake is very sweet and taste chocolate.",
        "allowed_content_compliance": ["complete", "partial", "incomplete"],
    }
}

VOCABULARY_MODE_SPECS = {
    "vocabulary_definition": {
        "family": "vocabulary",
        "schema_type": "definition_multiple_choice",
        "exercise_type": "definition_multiple_choice",
        "question_count": 5,
        "options_count": 3,
    },
    "vocabulary_match": {
        "family": "vocabulary",
        "schema_type": "matching",
        "exercise_type": "matching",
        "topic": "IT",
        "word_count": 8,
        "translation_count": 8,
    },
    "vocabulary_synonym": {
        "family": "vocabulary",
        "schema_type": "synonym_antonym",
        "exercise_type": "synonym_antonym",
        "expected_relation": "synonym",
        "question_count": 5,
        "options_count": 3,
    },
    "vocabulary_antonym": {
        "family": "vocabulary",
        "schema_type": "synonym_antonym",
        "exercise_type": "synonym_antonym",
        "expected_relation": "antonym",
        "question_count": 5,
        "options_count": 3,
    },
}

ALL_MODE_SPECS = {}
ALL_MODE_SPECS.update(TEST_SECTION_SPECS)
ALL_MODE_SPECS.update(GRAMMAR_MODE_SPECS)
ALL_MODE_SPECS.update(WRITING_MODE_SPECS)
ALL_MODE_SPECS.update(VOCABULARY_MODE_SPECS)


def get_mode_spec(mode_id: str) -> dict | None:
    return ALL_MODE_SPECS.get(mode_id)


def is_test_mode(mode_id: str) -> bool:
    spec = get_mode_spec(mode_id)
    return spec is not None and spec.get("family") == "test"


def is_grammar_mode(mode_id: str) -> bool:
    spec = get_mode_spec(mode_id)
    return spec is not None and spec.get("family") == "grammar"


def is_writing_mode(mode_id: str) -> bool:
    spec = get_mode_spec(mode_id)
    return spec is not None and spec.get("family") == "writing"


def is_vocabulary_mode(mode_id: str) -> bool:
    spec = get_mode_spec(mode_id)
    return spec is not None and spec.get("family") == "vocabulary"