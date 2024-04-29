from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import json


def evaluate_truthfulqa(output, answer):
    rouge = Rouge()
    bleu = 0
    rouge_l = 0
    rouge_1 = 0
    rouge_2 = 0
    for i in range(len(output)):
        if output[i]:
            bleu += sentence_bleu([answer[i].split()], output[i].split())
            rouge_l += rouge.get_scores(output[i], answer[i])[0]["rouge-l"]["f"]
            rouge_1 += rouge.get_scores(output[i], answer[i])[0]["rouge-1"]["f"]
            rouge_2 += rouge.get_scores(output[i], answer[i])[0]["rouge-2"]["f"]
    bleu /= len(output)
    rouge_l /= len(output)
    rouge_1 /= len(output)
    rouge_2 /= len(output)
    return bleu, rouge_l, rouge_1, rouge_2


if __name__ == "__main__":
    with open(
        "work_dirs/lit_llama_inference/truthful_qa_generation_peft.json",
        "r",
        encoding="utf-8",
    ) as f:
        result = json.load(f)
    answer = result["answer"]
    output = result["output"]
    bleu, rouge_l, rouge_1, rouge_2 = evaluate_truthfulqa(output, answer)
    print("-" * 30 + "PEFT" + "-" * 30)
    print(f"BLEU: {bleu}")
    print(f"ROUGE-l: {rouge_l}")
    print(f"ROUGE-1: {rouge_1}")
    print(f"ROUGE-2: {rouge_2}")

    # from beautifultable import BeautifulTable

    # table = BeautifulTable(precision=8)
    # for r in [8, 32, 64, 128, 256]:
    #     with open(
    #         f"work_dirs/lit_llama_inference/lora_r{r}/truthful_qa_generation_peft.json",
    #         "r",
    #         encoding="utf-8",
    #     ) as f:
    #         result = json.load(f)
    #     answer = result["answer"]
    #     output = result["output"]
    #     bleu, rouge_l, rouge_1, rouge_2 = evaluate_truthfulqa(output, answer)
    #     print("-" * 30 + "PEFT finetuned" + "-" * 30)
    #     print(f"BLEU: {bleu}")
    #     print(f"ROUGE-l: {rouge_l}")
    #     print(f"ROUGE-1: {rouge_1}")
    #     print(f"ROUGE-2: {rouge_2}")

    #     table.rows.append([bleu, rouge_l, rouge_1, rouge_2])

    # for layer in [3, 16]:
    #     with open(
    #         f"work_dirs/lit_llama_inference/top{layer}layernorm/truthful_qa_generation_peft.json",
    #         "r",
    #         encoding="utf-8",
    #     ) as f:
    #         result = json.load(f)
    #     answer = result["answer"]
    #     output = result["output"]
    #     bleu, rouge_l, rouge_1, rouge_2 = evaluate_truthfulqa(output, answer)
    #     print("-" * 30 + "PEFT finetuned" + "-" * 30)
    #     print(f"BLEU: {bleu}")
    #     print(f"ROUGE-l: {rouge_l}")
    #     print(f"ROUGE-1: {rouge_1}")
    #     print(f"ROUGE-2: {rouge_2}")

    #     table.rows.append([bleu, rouge_l, rouge_1, rouge_2])

    # table.columns.header = ["BLEU", "ROUGE-l", "ROUGE-1", "ROUGE-2"]
    # table.rows.header = ["r=8", "r=32", "r=64", "r=128", "r=256", "topk=3", "topk=16"]
    # table.set_style(BeautifulTable.STYLE_RST)
    # print(table)
    # with open("work_dirs/lit_llama_inference/truthfulqa_metric.txt", "w") as f:
    #     f.write(table.__str__())
