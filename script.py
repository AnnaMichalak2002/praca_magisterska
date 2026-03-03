import requests
import json
import time
from pathlib import Path
import sys

OLLAMA_URL_prompt = "http://localhost:11434/api/generate"
OLLAMA_URL_message = "http://localhost:11434/api/chat"

MODELS = ["llama3.1:8b", "qwen3:14b"]

def safe_filename(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")

def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def save_result(data, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def build_prompt(system: str, user: str) -> str:
    parts = [
        "SYSTEM:",
        system.strip(),
        "",
        "USER:",
        user.strip(),
    ]
    return "\n".join(parts)

def query_model_2(model: str, system_prompt: str, user_prompt: str):
    prompt = build_prompt(system_prompt, user_prompt)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    start = time.perf_counter()
    response = requests.post(OLLAMA_URL_prompt, json=payload)
    duration = round(time.perf_counter() - start, 3)

    response.raise_for_status()

    data = response.json()
    raw_output = data.get("response", "")

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        raise ValueError("Model did not return valid JSON")

    return {
        "model": model,
        "time": duration,
        "response": parsed
    }


def query_model(model: str, system_prompt: str, user_prompt: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False
    }

    start = time.perf_counter()
    response = requests.post(OLLAMA_URL_message, json=payload)
    duration = round(time.perf_counter() - start, 3)

    response.raise_for_status()

    data = response.json()

    raw_output = data.get("message", {}).get("content", "")

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        print("RAW OUTPUT:")
        print(raw_output)
        raise ValueError("Model did not return valid JSON")

    return {
        "model": model,
        "time": duration,
        "response": parsed
    }

if __name__ == "__main__":


    model = "llama3.1:8b"
    #model = "qwen3:14b"
    system_prompt = load_prompt("prompts/vocabulary/system_vocabulary.txt")
    user_prompt1 = load_prompt("prompts/vocabulary/user_vocabulary_match.txt")
    user_prompt = user_prompt1.replace("{topic}", "IT")

    safe_model = safe_filename(model)

    result2 = query_model_2(model, system_prompt, user_prompt)
    print("Czas:", result2["time"], "s")
    print(result2["response"])
    save_result(result2, f"results/vocabulary/generate/vocabulary_match_{safe_model}.json")

    result = query_model(model, system_prompt, user_prompt)
    print("Czas:", result["time"], "s")
    print(result["response"])

    
    save_result(result, f"results/vocabulary/chat/vocabulary_match_{safe_model}.json")
    sys.exit()
    # #model = "llama3.1:8b"-
    # model = "qwen3:14b"

    # system_prompt = load_prompt("prompts/writing/system_writing2.txt")
    # user_prompt = load_prompt("prompts/writing/user_writing.txt")

    # result = query_model(model, system_prompt, user_prompt)

    # print("Czas:", result["time"], "s")
    # print(result["response"])

    # safe_model = safe_filename(model)
    # save_result(result, f"results/writing/chat/writing_{safe_model}.json")

    # result2 = query_model_2(model, system_prompt, user_prompt)

    # print("Czas:", result2["time"], "s")
    # print(result2["response"])

    # safe_model = safe_filename(model)
    # save_result(result2, f"results/writing/generate/writing_{safe_model}.json")

    # sys.exit()
    # #model = "llama3.1:8b"
    # model = "qwen3:14b"

    # system_prompt = load_prompt("prompts/grammar/system_grammar.txt")
    # user_prompt = load_prompt("prompts/grammar/user_grammar_simple_vs_continuous.txt")

    # result = query_model(model, system_prompt, user_prompt)

    # print("Czas:", result["time"], "s")
    # print(result["response"])

    # safe_model = safe_filename(model)
    # save_result(result, f"results/grammar/chat/grammar_simple_vs_continuous_{safe_model}.json")

    # result2 = query_model_2(model, system_prompt, user_prompt)

    # print("Czas:", result2["time"], "s")
    # print(result2["response"])

    # safe_model = safe_filename(model)
    # save_result(result2, f"results/grammar/generate/grammar_simple_vs_continuous_{safe_model}.json")

    # sys.exit()

    # #model = "llama3.1:8b"
    # model = "qwen3:14b"

    # system_prompt = load_prompt("prompts/grammar/system_grammar.txt")
    # user_prompt = load_prompt("prompts/grammar/user_grammar_gerund_vs_infinitive.txt")

    # result = query_model(model, system_prompt, user_prompt)

    # print("Czas:", result["time"], "s")
    # print(result["response"])

    # safe_model = safe_filename(model)
    # save_result(result, f"results/grammar/chat/grammar_gerund_vs_infinitive_{safe_model}.json")

    # result2 = query_model_2(model, system_prompt, user_prompt)

    # print("Czas:", result2["time"], "s")
    # print(result2["response"])

    # safe_model = safe_filename(model)
    # save_result(result2, f"results/grammar/generate/grammar_gerund_vs_infinitive_{safe_model}.json")

    # sys.exit()
    # #model = "llama3.1:8b"
    # model = "qwen3:14b"

    # system_prompt = load_prompt("prompts/grammar/system_grammar.txt")
    # user_prompt = load_prompt("prompts/grammar/user_grammar_conditionals_1_2_3.txt")

    # result = query_model(model, system_prompt, user_prompt)

    # print("Czas:", result["time"], "s")
    # print(result["response"])

    # safe_model = safe_filename(model)
    # save_result(result, f"results/grammar/chat/grammar_conditionals_1_2_3_{safe_model}.json")

    # result2 = query_model_2(model, system_prompt, user_prompt)

    # print("Czas:", result2["time"], "s")
    # print(result2["response"])

    # safe_model = safe_filename(model)
    # save_result(result2, f"results/grammar/generate/grammar_conditionals_1_2_3_{safe_model}.json")

    # sys.exit()
    # #model = "llama3.1:8b"
    # model = "qwen3:14b"

    # system_prompt = load_prompt("prompts/grammar/system_grammar.txt")
    # user_prompt = load_prompt("prompts/grammar/user_grammar_past_vs_present_perfect.txt")

    # result = query_model(model, system_prompt, user_prompt)

    # print("Czas:", result["time"], "s")
    # print(result["response"])

    # safe_model = safe_filename(model)
    # save_result(result, f"results/grammar/chat/grammar_past_vs_present_perfect_{safe_model}.json")

    # result2 = query_model_2(model, system_prompt, user_prompt)

    # print("Czas:", result2["time"], "s")
    # print(result2["response"])

    # safe_model = safe_filename(model)
    # save_result(result2, f"results/grammar/generate/grammar_past_vs_present_perfect_{safe_model}.json")

    # sys.exit()
    # #model = "llama3.1:8b"
    # model = "qwen3:14b"

    # system_prompt = load_prompt("prompts/vocabulary/system_vocabulary.txt")
    # user_prompt = load_prompt("prompts/vocabulary/user_vocabulary_definition.txt")

    # result = query_model(model, system_prompt, user_prompt)

    # print("Czas:", result["time"], "s")
    # print(result["response"])

    # safe_model = safe_filename(model)
    # save_result(result, f"results/vocabulary/chat/vocabulary_definition_{safe_model}.json")

    # result2 = query_model_2(model, system_prompt, user_prompt)

    # print("Czas:", result2["time"], "s")
    # print(result2["response"])

    # safe_model = safe_filename(model)
    # save_result(result2, f"results/vocabulary/generate/vocabulary_definition_{safe_model}.json")


    # sys.exit()
    # #model = "llama3.1:8b"
    # model = "qwen3:14b"
    
    # #relation = "synonym"
    # relation = "antonym"
    # system_prompt = load_prompt("prompts/vocabulary/system_vocabulary.txt")
    # template = load_prompt("prompts/vocabulary/user_vocabulary_synonym_antonym.txt")
    # user_prompt = template.replace("{relation}", relation)
    
    # safe_model = safe_filename(model)

    # result2 = query_model_2(model, system_prompt, user_prompt)
    # print("Czas:", result2["time"], "s")
    # print(result2["response"])
    # save_result(result2, f"results/vocabulary/generate/vocabulary_{relation}_{safe_model}.json")

    # result = query_model(model, system_prompt, user_prompt)
    # print("Czas:", result["time"], "s")
    # print(result["response"])

    
    # save_result(result, f"results/vocabulary/chat/vocabulary_{relation}_{safe_model}.json")



    # sys.exit()

    # model = "llama3.1:8b"
    # model = "qwen3:14b"
    # system_prompt = load_prompt("prompts/vocabulary/system_vocabulary.txt")
    # user_prompt = load_prompt("prompts/vocabulary/user_vocabulary_match.txt")
    
    # safe_model = safe_filename(model)

    # result2 = query_model_2(model, system_prompt, user_prompt)
    # print("Czas:", result2["time"], "s")
    # print(result2["response"])
    # save_result(result2, f"results/vocabulary/generate/vocabulary_match_{safe_model}.json")

    # result = query_model(model, system_prompt, user_prompt)
    # print("Czas:", result["time"], "s")
    # print(result["response"])

    
    # save_result(result, f"results/vocabulary/chat/vocabulary_match_{safe_model}.json")