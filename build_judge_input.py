from pathlib import Path
import json

RESULTS_ROOT = Path("results")
OUTPUT_DIR = Path("judge_pipeline")
OUTPUT_JSONL = OUTPUT_DIR / "judge_input.jsonl"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_task_family(mode_id: str) -> str:
    if mode_id.startswith("test_section_"):
        return "test"
    if mode_id.startswith("grammar_"):
        return "grammar"
    if mode_id == "writing":
        return "writing"
    if mode_id.startswith("vocabulary_"):
        return "vocabulary"
    return "unknown"


def load_prompt_if_exists(path_str: str | None) -> str | None:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        return None

    return path.read_text(encoding="utf-8")


def build_validation_index(validation_file: Path) -> dict:
    """
    Builds index:
    attempt_file_name -> validation_record
    """
    validation_items = load_json(validation_file)

    index = {}
    if isinstance(validation_items, list):
        for item in validation_items:
            attempt_file = item.get("attempt_file")
            if attempt_file:
                index[attempt_file] = item

    return index


def build_record(attempt_data: dict, validation_item: dict, attempt_file: Path) -> dict:
    mode_id = attempt_data.get("mode_id")
    model = attempt_data.get("model")
    attempt = attempt_data.get("attempt")

    safe_model_dir = attempt_file.parent.name
    record_id = f"{mode_id}__{safe_model_dir}__attempt_{attempt:03d}"

    system_prompt = load_prompt_if_exists(attempt_data.get("system_prompt_path"))
    user_prompt = load_prompt_if_exists(attempt_data.get("user_prompt_path"))

    record = {
        "record_id": record_id,
        "model": model,
        "mode_id": mode_id,
        "task_family": detect_task_family(mode_id),
        "attempt": attempt,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": attempt_data.get("response"),
        "source_file": str(attempt_file.as_posix()),
        "validation": {
            "recoverable": validation_item.get("recoverable"),
            "non_fatal_validation_errors": validation_item.get("non_fatal_validation_errors", []),
}
    }

    return record


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_missing_validation_item = 0
    skipped_not_tolerant_valid = 0
    skipped_missing_response = 0

    with OUTPUT_JSONL.open("w", encoding="utf-8") as out_f:
        for validation_file in sorted(RESULTS_ROOT.rglob("validation_tolerant.json")):
            result_dir = validation_file.parent
            validation_index = build_validation_index(validation_file)

            for attempt_file in sorted(result_dir.glob("attempt_*.json")):
                validation_item = validation_index.get(attempt_file.name)
                if validation_item is None:
                    skipped_missing_validation_item += 1
                    continue

                if validation_item.get("tolerant_valid") is not True:
                    skipped_not_tolerant_valid += 1
                    continue

                attempt_data = load_json(attempt_file)

                if attempt_data.get("response") is None:
                    skipped_missing_response += 1
                    continue

                record = build_record(
                    attempt_data=attempt_data,
                    validation_item=validation_item,
                    attempt_file=attempt_file,
                )

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Saved: {OUTPUT_JSONL}")
    print(f"Written records: {written}")
    print(f"Skipped not tolerant_valid: {skipped_not_tolerant_valid}")


if __name__ == "__main__":
    main()