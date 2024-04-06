import re
import json


def preprocess(sentence):
    match = re.search(r"[A-D]", sentence)
    if match:
        sentence = match.group(0)
    else:
        sentence = ""
    return sentence


def evaluate_medmcqa(output, ground_truth):
    total = len(ground_truth)
    acc = 0
    for i in range(len(output)):
        output[i] = preprocess(output[i])
        ground_truth[i] = preprocess(ground_truth[i])
        if output[i] == ground_truth[i]:
            acc += 1
    return acc / total


if __name__ == "__main__":
    with open(
        "work_dirs/lit_llama_lora_inference/medmcqa.json", "r", encoding="utf-8"
    ) as f:
        result = json.load(f)
    answer = result["answer"]
    output = result["output"]
    accuracy = evaluate_medmcqa(output, answer)
    print(f"Accuracy: {accuracy}")
    