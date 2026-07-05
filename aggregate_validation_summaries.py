from pathlib import Path
import json
import pandas as pd


RESULTS_ROOT = Path("results")
OUTPUT_CSV = Path("aggregated/validation_summary_aggregated.csv")
OUTPUT_XLSX = Path("aggregated/validation_summary_aggregated.xlsx")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_main_key_from_counter_dict(data: dict | None) -> str | None:
    if not isinstance(data, dict) or not data:
        return None
    return max(data.items(), key=lambda kv: kv[1])[0]


def build_row(summary_path: Path) -> dict:
    summary = load_json(summary_path)

    # struktura:
    # results/{mode_id}/{safe_model}/summary_tolerant.json
    model_dir = summary_path.parent
    mode_dir = model_dir.parent

    mode_id = mode_dir.name
    model = model_dir.name

    fatal_errors = summary.get("fatal_error_counts_normalized", {}) or {}
    non_fatal_errors = summary.get("non_fatal_error_counts_normalized", {}) or {}
    not_validated_reasons = summary.get("not_validated_reasons", {}) or {}

    row = {
        "mode_id": mode_id,
        "model": model,

        "all_attempts": summary.get("all_attempts"),
        "validated_attempts": summary.get("validated_attempts"),
        "not_validated_attempts": summary.get("not_validated_attempts"),

        "strictly_valid_count": summary.get("strictly_valid_count"),
        "strictly_valid_rate": summary.get("strictly_valid_rate"),
        "strictly_valid_rate_over_all_attempts": summary.get("strictly_valid_rate_over_all_attempts"),

        "recovered_by_tolerant_count": summary.get("recovered_by_tolerant_count"),
        "recovered_by_tolerant_rate_over_validated": summary.get("recovered_by_tolerant_rate_over_validated"),
        "recovered_by_tolerant_rate_over_non_strict": summary.get("recovered_by_tolerant_rate_over_non_strict"),
        "recovered_by_tolerant_rate_over_all_attempts": summary.get("recovered_by_tolerant_rate_over_all_attempts"),

        "final_valid_count": summary.get("final_valid_count"),
        "final_valid_rate": summary.get("final_valid_rate"),
        "final_valid_rate_over_all_attempts": summary.get("final_valid_rate_over_all_attempts"),

        "final_invalid_count": summary.get("final_invalid_count"),
        "final_invalid_rate": summary.get("final_invalid_rate"),
        "final_invalid_rate_over_all_attempts": summary.get("final_invalid_rate_over_all_attempts"),

        "fatal_error_main_normalized": pick_main_key_from_counter_dict(fatal_errors),
        "fatal_error_unique_count": len(fatal_errors),
        "fatal_error_counts_normalized": json.dumps(fatal_errors, ensure_ascii=False),

        "non_fatal_error_main_normalized": pick_main_key_from_counter_dict(non_fatal_errors),
        "non_fatal_error_unique_count": len(non_fatal_errors),
        "non_fatal_error_counts_normalized": json.dumps(non_fatal_errors, ensure_ascii=False),
    }

    return row


def main():
    summary_paths = sorted(RESULTS_ROOT.rglob("summary_tolerant.json"))

    if not summary_paths:
        print("No summary_tolerant.json files found.")
        return

    rows = [build_row(path) for path in summary_paths]
    df = pd.DataFrame(rows)

    df = df.sort_values(by=["mode_id", "model"]).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df.to_excel(OUTPUT_XLSX, index=False)

    print(f"Saved CSV:  {OUTPUT_CSV}")
    print(f"Saved XLSX: {OUTPUT_XLSX}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()