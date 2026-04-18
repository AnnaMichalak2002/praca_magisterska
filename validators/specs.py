from typing import Any


TEST_SECTION_SPECS = {
    "test_section_1": {
        "section_id": 1,
        "section_title": "Grammar and Vocabulary (A1-A2)",
        "cefr_range": "A1-A2",
        "question_count": 10,
        "question_numbers": list(range(1, 11)),
        "requires_reading_text": False,
    },
    "test_section_2": {
        "section_id": 2,
        "section_title": "Grammar and Vocabulary (B1)",
        "cefr_range": "B1",
        "question_count": 10,
        "question_numbers": list(range(11, 21)),
        "requires_reading_text": False,
    },
    "test_section_3": {
        "section_id": 3,
        "section_title": "Grammar and Vocabulary (B2)",
        "cefr_range": "B2",
        "question_count": 10,
        "question_numbers": list(range(21, 31)),
        "requires_reading_text": False,
    },
    "test_section_4": {
        "section_id": 4,
        "section_title": "Advanced Grammar and Structures (C1-C2)",
        "cefr_range": "C1-C2",
        "question_count": 5,
        "question_numbers": list(range(31, 36)),
        "requires_reading_text": False,
    },
    "test_section_5": {
        "section_id": 5,
        "section_title": "Reading Comprehension",
        "cefr_range": "B2-C1",
        "question_count": 5,
        "question_numbers": list(range(36, 41)),
        "requires_reading_text": True,
    },
    "test_section_6": {
        "section_id": 6,
        "section_title": "Language Functions",
        "cefr_range": "B1-B2",
        "question_count": 5,
        "question_numbers": list(range(41, 46)),
        "requires_reading_text": False,
    },
    "test_section_7": {
        "section_id": 7,
        "section_title": "Advanced Vocabulary",
        "cefr_range": "B2-C1",
        "question_count": 5,
        "question_numbers": list(range(46, 51)),
        "requires_reading_text": False,
    },
}


def is_test_mode(mode_id: str) -> bool:
    return mode_id in TEST_SECTION_SPECS