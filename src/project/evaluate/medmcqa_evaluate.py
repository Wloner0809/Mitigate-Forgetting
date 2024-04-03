import re


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
        "/home/terencewang/UNC-Project/work_dirs/lit_llama_lora_inference/predict_result.txt",
        "r",
    ) as f:
        output = f.readlines()
    output = [x.strip() for x in output]
    with open(
        "/home/terencewang/UNC-Project/work_dirs/lit_llama_lora_inference/truth.txt",
        "r",
    ) as f:
        ground_truth = f.readlines()
    ground_truth = [x.strip() for x in ground_truth]
    print(evaluate_medmcqa(output, ground_truth))
