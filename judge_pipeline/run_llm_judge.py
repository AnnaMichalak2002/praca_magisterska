from pathlib import Path
import json
import time
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


INPUT_JSONL = Path("judge_pipeline/judge_input.jsonl")
OUTPUT_JSONL = Path("judge_pipeline/judge_results.jsonl")
LOG_FILE = Path("judge_pipeline/run_llm_judge.log")

MODEL_NAME = "Qwen/Qwen3.6-27B"

MAX_NEW_TOKENS = 1000
RESUME = True

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated materials for learning English as a foreign language.

Evaluate each response as a teacher/expert assessing whether the material is suitable, correct, and ready to be given to a learner without correction.

Base your evaluation only on:

* the system prompt,
* the user prompt,
* the model response.

Ignore hidden technical issues, validator metadata, and non-fatal schema problems that are invisible to the learner. Evaluate only the visible educational content.

A response may be marked usable_for_learner = "yes" only if it could be safely given to a learner without correction.
A single serious factual, linguistic, or pedagogical error may be enough to make the material unusable.

Do not include reasoning or analysis outside the required JSON.

Evaluate the response using these criteria:

1. instruction_compliance_score (1-5)
   How well the response follows the user’s instruction and fulfills the requested task.
   Consider whether it matches the requested exercise type, topic, scope, constraints, and required content.
   1 = very poor, 2 = poor, 3 = partial, 4 = good, 5 = very good.

2. factual_correctness_score (1-5)
   How correct the content is from a language-learning perspective.
   Consider correctness of answer keys, corrections, explanations, grammar content, vocabulary content, and language-related claims.
   Penalize incorrect answer keys, misleading explanations, wrong corrections, wrong topic coverage, wrong level or content that could teach the learner something incorrect.
   1 = very poor, 2 = poor, 3 = partly correct, 4 = mostly correct, 5 = very correct.

3. didactic_value_score (1-5)
   How useful the response is for learning.
   Consider clarity, adequacy of level, usefulness of examples, quality of distractors, usefulness of explanations, and overall educational value.
   1 = very poor, 2 = poor, 3 = moderate, 4 = good, 5 = very good.

Then decide:

* usable_for_learner = "yes" only if the full material is suitable for learner use without correction,
* usable_for_learner = "no" if the response is misleading, substantially incorrect, clearly incomplete, ambiguous in a harmful way, off-topic, too easy or too hard in a serious way, or educationally poor.

Return only valid JSON.
Do not return markdown.
Do not return code fences.
Do not return any text outside JSON.

The JSON must have exactly these fields:
{
"instruction_compliance_score": "<integer>",
"factual_correctness_score": "<integer>",
"didactic_value_score": "<integer>",
"usable_for_learner": "<string>",
"critical_issues": ["<string>"],
"short_explanation": "<string>"
}

Rules:

* score fields must be JSON numbers, not strings
* instruction_compliance_score must be an integer from 1 to 5
* factual_correctness_score must be an integer from 1 to 5
* didactic_value_score must be an integer from 1 to 5
* usable_for_learner must be exactly "yes" or "no"
* critical_issues must be an array of concise strings; use [] if there are no major issues
* critical_issues must contain at most 3 items
* each critical issue should describe one distinct major problem
* short_explanation must be concise, specific, and based only on the visible response
* short_explanation must be 1-3 concise sentences
* Be strict but fair

"""
WRITING_RULES = """Task-specific rules for writing:
- Check whether the correction, error identification, explanations, and feedback are accurate and pedagogically safe.
- If the corrected text contains an important linguistic error, this is a serious problem.
- If the feedback incorrectly claims that required content is missing when it is actually present, this is a serious problem.
- If the model gives false, misleading, or linguistically incorrect explanations, this is a serious problem.
- If the model misclassifies important learner errors in a way that could mislead the learner, this is a serious problem.
- The corrected text should improve the learner’s text without changing the intended meaning unnecessarily.
- The response must match the requested task and content requirements.
- Mark usable_for_learner as "yes" only if the learner could safely use the correction and explanations without teacher correction.
"""

GRAMMAR_RULES = """Task-specific rules for grammar:
- The exercise must match the requested grammar topic exactly.
- If the task is about specific grammar forms or tenses, introducing another tense, another structure, or mixed forms not requested is a serious problem.
- The target level must match the requested level closely enough to be usable for the intended learner.
- The answer key must be correct and unambiguous.
- If more than one option is plausibly correct, this is a serious problem.
- If the marked answer is incorrect, this is a serious problem.
- Sentences must be grammatically meaningful and suitable for the target level.
- The material must not teach incorrect grammar patterns.
- Mark usable_for_learner as "yes" only if the full exercise could be given to a learner without correcting any item.
"""

VOCABULARY_RULES = """Task-specific rules for vocabulary:
- The exercise must match the requested vocabulary task type exactly.
- The content must match the requested topic and level when such constraints are given.
- In matching tasks, every pair must be correct. One clearly incorrect or hallucinated pair is a serious problem.
- In synonym, antonym, definition, and multiple-choice vocabulary tasks, the correct answer must be unambiguous.
- If more than one option is plausibly correct, this is a serious problem.
- If the marked answer is incorrect, this is a serious problem.
- Distractors should be plausible but clearly wrong.
- The material must not teach incorrect meanings, false translations, or misleading vocabulary relationships.
- Mark usable_for_learner as "yes" only if the learner could use the full material without correction.
"""

TEST_RULES = """Task-specific rules for tests:
- The exercise must match the requested test section type, scope, and CEFR level.
- Questions must be appropriate for the intended level and section purpose.
- The correct answer must be unambiguous.
- If more than one option is plausibly correct, this is a serious problem.
- If the marked answer is incorrect, this is a serious problem.
- Distractors should be plausible and test language competence, not arbitrary preference or unclear context.
- Questions must not rely on missing context that makes the answer subjective or debatable.
- The material must not mislead the learner about correct language use or test expectations.
- Mark usable_for_learner as "yes" only if the full section could be given to a learner without correction.
"""

def build_task_specific_rules(record: dict) -> str:
    task_family = record.get("task_family", "")

    if task_family == "writing":
        return WRITING_RULES
    if task_family == "grammar":
        return GRAMMAR_RULES
    if task_family == "vocabulary":
        return VOCABULARY_RULES
    if task_family == "test":
        return TEST_RULES

    return ""

def log_message(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{timestamp} | {message}"
    print(full_message, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(full_message + "\n")


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    record_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                record_id = data.get("record_id")
                if isinstance(record_id, str):
                    record_ids.add(record_id)
            except Exception:
                continue
    return record_ids


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    raise ValueError("No complete JSON object found")


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_user_prompt(record: dict) -> str:
    task_specific_rules = build_task_specific_rules(record)

    return f"""Evaluate the following record.

task_family: {record["task_family"]}
mode_id: {record["mode_id"]}

TASK-SPECIFIC RULES:
{task_specific_rules}

SYSTEM PROMPT:
{record["system_prompt"]}

USER PROMPT:
{record["user_prompt"]}

MODEL RESPONSE:
{pretty_json(record["response"])}

Return only valid JSON with the required fields."""


def build_messages(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record)},
    ]


def parse_judge_output(raw_text: str) -> dict:
    json_text = extract_first_json_object(raw_text.strip())
    parsed = json.loads(json_text)

    required_fields = {
        "instruction_compliance_score",
        "factual_correctness_score",
        "didactic_value_score",
        "usable_for_learner",
        "critical_issues",
        "short_explanation",
    }

    if set(parsed.keys()) != required_fields:
        raise ValueError(f"Unexpected JSON fields: {sorted(parsed.keys())}")

    for field in [
        "instruction_compliance_score",
        "factual_correctness_score",
        "didactic_value_score",
    ]:
        value = parsed[field]

        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                value = int(value)

        if not isinstance(value, int) or not (1 <= value <= 5):
            raise ValueError(f"{field} must be integer 1-5, got: {parsed[field]!r}")

        parsed[field] = value

    if parsed["usable_for_learner"] not in {"yes", "no"}:
        raise ValueError(f'usable_for_learner must be "yes" or "no", got: {parsed["usable_for_learner"]}')

    if not isinstance(parsed["critical_issues"], list) or not all(isinstance(x, str) for x in parsed["critical_issues"]):
        raise ValueError("critical_issues must be list[str]")

    if not isinstance(parsed["short_explanation"], str) or not parsed["short_explanation"].strip():
        raise ValueError("short_explanation must be non-empty string")

    return parsed


def generate_one(model, tokenizer, record: dict) -> tuple[dict, str]:
    messages = build_messages(record)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    try:
        parsed = parse_judge_output(raw_text)
    except Exception as e:
        raise ValueError(f"{e} | raw_text_preview={raw_text[:1000]!r}")

    return parsed


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    log_message("=== JUDGE RUN STARTED ===")
    log_message(f"Model: {MODEL_NAME}")

    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    records = load_jsonl(INPUT_JSONL)
    log_message(f"Loaded input records: {len(records)}")

    done_ids = load_existing_record_ids(OUTPUT_JSONL) if RESUME else set()
    if RESUME:
        log_message(f"Existing output records: {len(done_ids)}")

    log_message("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    log_message("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processed = 0
    skipped = 0
    failed = 0

    for idx, record in enumerate(records, start=1):
        record_id = record["record_id"]

        if RESUME and record_id in done_ids:
            skipped += 1
            continue

        log_message(f"RUN | {idx}/{len(records)} | {record_id}")

        started = time.perf_counter()

        try:
            parsed = generate_one(model, tokenizer, record)
            elapsed = round(time.perf_counter() - started, 3)

            result = {
                "record_id": record_id,
                "judge_model": MODEL_NAME,
                "task_family": record.get("task_family"),
                "mode_id": record.get("mode_id"),
                "model_under_evaluation": record.get("model"),
                "attempt": record.get("attempt"),
                "evaluation_time_seconds": elapsed,
                "judge_result": parsed,
                "status": "ok",
            }

            append_jsonl(OUTPUT_JSONL, result)
            processed += 1

        except Exception as e:
            elapsed = round(time.perf_counter() - started, 3)

            result = {
                "record_id": record_id,
                "judge_model": MODEL_NAME,
                "task_family": record.get("task_family"),
                "mode_id": record.get("mode_id"),
                "model_under_evaluation": record.get("model"),
                "attempt": record.get("attempt"),
                "evaluation_time_seconds": elapsed,
                "judge_result": None,
                "status": "error",
                "error": str(e),
            }

            append_jsonl(OUTPUT_JSONL, result)
            failed += 1
            log_message(f"ERROR | {record_id} | {e}")

    log_message(f"Processed: {processed}")
    log_message(f"Skipped:   {skipped}")
    log_message(f"Failed:    {failed}")
    log_message("=== JUDGE RUN FINISHED ===")


if __name__ == "__main__":
    main()