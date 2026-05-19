import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from agent import DataAnalystAgent, RequestCategory

_raw = json.load(open(os.path.join(os.path.dirname(__file__), "eval_classify.json")))
EVALS: list[tuple[str, RequestCategory]] = [
    (entry["question"], RequestCategory(entry["category"])) for entry in _raw
]


def run_single(agent: DataAnalystAgent, idx: int, question: str, expected: RequestCategory):
    predicted = agent._classify(question)
    return idx, question, expected, predicted, predicted == expected


def main():
    agent = DataAnalystAgent()

    results = [None] * len(EVALS)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(run_single, agent, i, q, e): i
            for i, (q, e) in enumerate(EVALS)
        }
        for future in as_completed(futures):
            idx, question, expected, predicted, correct = future.result()
            results[idx] = (question, expected, predicted, correct)
            status = "OK" if correct else "FAIL"
            print(f"[{status}] {expected.value:15} -> {predicted.value:15} | {question[:60]}")

    per_category: dict[RequestCategory, dict] = {c: {"correct": 0, "total": 0} for c in RequestCategory}
    for _, expected, predicted, correct in results:
        per_category[expected]["total"] += 1
        if correct:
            per_category[expected]["correct"] += 1

    total_correct = sum(r["correct"] for r in per_category.values())
    total = len(EVALS)

    print("\n--- Results ---")
    for cat, stats in per_category.items():
        acc = stats["correct"] / stats["total"] * 100
        print(f"{cat.value:15}: {stats['correct']}/{stats['total']} ({acc:.0f}%)")
    print(f"{'Overall':15}: {total_correct}/{total} ({total_correct / total * 100:.0f}%)")


if __name__ == "__main__":
    main()
