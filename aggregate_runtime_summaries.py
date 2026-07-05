from pathlib import Path
import json
import pandas as pd


RESULTS_ROOT = Path("results")
OUTPUT_CSV = Path("aggregated/runtime_summary_aggregated.csv")
OUTPUT_XLSX = Path("aggregated/runtime_summary_aggregated.xlsx")


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
    # results/{mode_id}/{safe_model}/summary_runtime.json
    model_dir = summary_path.parent
    mode_dir = model_dir.parent

    mode_id = mode_dir.name
    model = model_dir.name

    ollama_ps_snapshot = summary.get("ollama_ps_snapshot", {}) or {}
    postprocessing_success_only = summary.get("postprocessing_steps_counts_success_only", {}) or {}
    done_reason_success_only = summary.get("done_reason_counts_success_only", {}) or {}

    row = {
        "mode_id": mode_id,
        "model": model,

        "total_attempts": summary.get("total_attempts"),
        "successful_attempts": summary.get("successful_attempts"),
        "failed_attempts": summary.get("failed_attempts"),
        "success_rate": summary.get("success_rate"),

        "average_time_seconds_success_only": summary.get("average_time_seconds_success_only"),
        "min_time_seconds_success_only": summary.get("min_time_seconds_success_only"),
        "max_time_seconds_success_only": summary.get("max_time_seconds_success_only"),

        "avg_tokens_per_second_success_only": summary.get("avg_tokens_per_second_success_only"),
        "avg_total_duration_ns_success_only": summary.get("avg_total_duration_ns_success_only"),
        "avg_prompt_eval_count_success_only": summary.get("avg_prompt_eval_count_success_only"),
        "avg_eval_count_success_only": summary.get("avg_eval_count_success_only"),

        "avg_gpu_utilization_percent_success_only": summary.get("avg_gpu_utilization_percent_success_only"),
        "avg_gpu_memory_used_mib_success_only": summary.get("avg_gpu_memory_used_mib_success_only"),
        "avg_gpu_temperature_c_success_only": summary.get("avg_gpu_temperature_c_success_only"),
        "avg_gpu_power_draw_w_success_only": summary.get("avg_gpu_power_draw_w_success_only"),
        "avg_gpu_energy_wh_success_only": summary.get("avg_gpu_energy_wh_success_only"),

        "processor": ollama_ps_snapshot.get("processor"),
        "loaded_size": ollama_ps_snapshot.get("loaded_size"),
        "context": ollama_ps_snapshot.get("context"),

        "done_reason_main_success_only": pick_main_key_from_counter_dict(done_reason_success_only),
        "done_reason_success_only_unique_count": len(done_reason_success_only),
        "done_reason_success_only_counts": json.dumps(done_reason_success_only, ensure_ascii=False),

        "postprocessing_success_only_main": pick_main_key_from_counter_dict(postprocessing_success_only),
        "postprocessing_success_only_unique_count": len(postprocessing_success_only),
        "postprocessing_success_only_counts": json.dumps(postprocessing_success_only, ensure_ascii=False),
    }

    return row


def main():
    summary_paths = sorted(RESULTS_ROOT.rglob("summary_runtime.json"))

    if not summary_paths:
        print("No summary_runtime.json files found.")
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