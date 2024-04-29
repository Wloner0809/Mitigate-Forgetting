import re
import json
from matplotlib import pyplot as plt


def preprocess(sentence):
    match = re.search(r"[A-D]", sentence)
    if match:
        sentence = match.group(0)
    else:
        sentence = ""
    return sentence


def evaluate_medmcqa(output, ground_truth, path):
    total = len(ground_truth)
    acc = 0
    A, B, C, D = 0, 0, 0, 0
    A_truth, B_truth, C_truth, D_truth = 0, 0, 0, 0
    for i in range(len(output)):
        output[i] = preprocess(output[i])
        ground_truth[i] = preprocess(ground_truth[i])
        if output[i] == ground_truth[i]:
            acc += 1
        if output[i] == "A":
            A += 1
        elif output[i] == "B":
            B += 1
        elif output[i] == "C":
            C += 1
        elif output[i] == "D":
            D += 1
        if ground_truth[i] == "A":
            A_truth += 1
        elif ground_truth[i] == "B":
            B_truth += 1
        elif ground_truth[i] == "C":
            C_truth += 1
        elif ground_truth[i] == "D":
            D_truth += 1
    plt.figure(figsize=(6, 4))
    plt.title("MedMCQA Evaluation", loc="center", fontweight="bold")
    plt.xlabel("Options", loc="center", fontweight="bold")
    plt.ylabel("Count", loc="center", fontweight="bold")
    plt.bar(
        [i for i in range(4)],
        [A, B, C, D],
        label="Output",
        width=0.2,
        color="#f9766e",
        edgecolor="k",
        zorder=2,
    )
    plt.bar(
        [i + 0.2 for i in range(4)],
        [A_truth, B_truth, C_truth, D_truth],
        label="Ground Truth",
        width=0.2,
        color="#00bfc4",
        edgecolor="k",
        zorder=2,
        hatch="/",
    )
    plt.xticks([i + 0.1 for i in range(4)], ["A", "B", "C", "D"], fontweight="bold")
    plt.yticks(fontweight="bold")
    plt.legend(frameon=False, loc="best")
    for a, b, i in zip([A, B, C, D], [A_truth, B_truth, C_truth, D_truth], range(4)):
        plt.text(i, a, str(a), ha="center", va="bottom", fontsize=10, fontweight="bold")
        plt.text(
            i + 0.2, b, str(b), ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    plt.grid(ls="--", alpha=0.8)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linestyle("-")
    plt.gca().spines["left"].set_linewidth(2.5)
    plt.gca().spines["bottom"].set_linestyle("-")
    plt.gca().spines["bottom"].set_linewidth(2.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(path)
    plt.clf()
    return acc / total


if __name__ == "__main__":
    # from beautifultable import BeautifulTable

    # table = BeautifulTable()
    # for r in [8, 32, 64, 128, 256]:
    #     with open(
    #         f"work_dirs/lit_llama_inference/lora_r{r}/medmcqa_peft.json",
    #         "r",
    #         encoding="utf-8",
    #     ) as f:
    #         result = json.load(f)
    #     answer = result["answer"]
    #     output = result["output"]
    #     accuracy = evaluate_medmcqa(
    #         output, answer, f"work_dirs/lit_llama_inference/lora_r{r}/medmcqa_peft.png"
    #     )
    #     print(f"PEFT finetuned Accuracy: {accuracy}")

    #     table.rows.append([accuracy])

    # for layer in [3, 16]:
    #     with open(
    #         f"work_dirs/lit_llama_inference/top{layer}layernorm/medmcqa_peft.json",
    #         "r",
    #         encoding="utf-8",
    #     ) as f:
    #         result = json.load(f)
    #     answer = result["answer"]
    #     output = result["output"]
    #     accuracy = evaluate_medmcqa(
    #         output,
    #         answer,
    #         f"work_dirs/lit_llama_inference/top{layer}layernorm/medmcqa_peft.png",
    #     )
    #     print(f"PEFT finetuned Accuracy: {accuracy}")

    #     table.rows.append([accuracy])

    # table.columns.header = ["acc"]
    # table.rows.header = ["r=8", "r=32", "r=64", "r=128", "r=256", "topk=3", "topk=16"]
    # table.set_style(BeautifulTable.STYLE_RST)
    # print(table)
    # with open("work_dirs/lit_llama_inference/medmcqa_acc.txt", "w") as f:
    #     f.write(table.__str__())

    with open(
        "work_dirs/lit_llama_inference/medmcqa_peft.json",
        "r",
        encoding="utf-8",
    ) as f:
        result = json.load(f)
    answer = result["answer"]
    output = result["output"]
    accuracy = evaluate_medmcqa(
        output, answer, "work_dirs/lit_llama_inference/medmcqa_peft.png"
    )
    print(f"PEFT Accuracy: {accuracy}")
