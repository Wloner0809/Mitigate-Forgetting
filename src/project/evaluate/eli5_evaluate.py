from bert_score import score
import json


def evalute_eli5(output, answer):
    P, R, F1 = score(
        output,
        answer,
        model_type="bert-base-uncased",
        lang="en",
        rescale_with_baseline=True,
    )
    return P.mean(), R.mean(), F1.mean()


if __name__ == "__main__":
    with open(
        "work_dirs/lit_llama_inference/eli5_category_baseline.json",
        "r",
        encoding="utf-8",
    ) as f:
        result = json.load(f)
    answer = result["answer"]
    output = result["output"]
    P, R, F1 = evalute_eli5(output, answer)
    print("-" * 30 + "Baseline" + "-" * 30)
    print(f"P: {P}")
    print(f"R: {R}")
    print(f"F1: {F1}")

    with open(
        "work_dirs/lit_llama_inference/eli5_category_peft.json",
        "r",
        encoding="utf-8",
    ) as f:
        result = json.load(f)
    answer = result["answer"]
    output = result["output"]
    P, R, F1 = evalute_eli5(output, answer)
    print("-" * 30 + "Peft" + "-" * 30)
    print(f"P: {P}")
    print(f"R: {R}")
    print(f"F1: {F1}")

    with open(
        "work_dirs/lit_llama_inference/eli5_category_freeze.json",
        "r",
        encoding="utf-8",
    ) as f:
        result = json.load(f)
    answer = result["answer"]
    output = result["output"]
    P, R, F1 = evalute_eli5(output, answer)
    print("-" * 30 + "Freeze" + "-" * 30)
    print(f"P: {P}")
    print(f"R: {R}")
    print(f"F1: {F1}")
