from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.prompter import Prompter


class MedmcqaDataset(Dataset):
    # split the dataset into train, valid, test with ratio 8:1:1
    SAMPLE_NUM = {"train": 16000, "valid": 2000, "test": 2000}

    def __init__(self, subset) -> None:
        super().__init__()
        self.subset = subset
        self.data_path = "/data/terencewang/medmcqa/data"
        self.tokenizer_path = "/data/terencewang/llama2-hf"
        self.length = self.SAMPLE_NUM[subset]
        self.data = self.prepare_data()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "input_ids": self.data["input_ids"][index],
            "attention_mask": self.data["attention_mask"][index],
            "labels": self.data["labels"][index],
            "index": index,
        }

    def prepare_data(self):
        dataset = load_dataset(self.data_path)
        train_data = dataset["train"].filter(lambda x: x["choice_type"] == "single")
        train_data = train_data.map(
            lambda x: self._preprocess(x),
            remove_columns=[
                "question",
                "opa",
                "opb",
                "opc",
                "opd",
                "cop",
                "id",
                "choice_type",
                "exp",
                "subject_name",
                "topic_name",
            ],
        )
        all_data = train_data.select(
            range(
                self.SAMPLE_NUM["train"]
                + self.SAMPLE_NUM["valid"]
                + self.SAMPLE_NUM["test"]
            )
        )

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        all_data = all_data.map(
            lambda x: self._tokenized_prompt(tokenizer, x),
            remove_columns=["instruction", "input", "output"],
        )
        all_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return all_data

    def _preprocess(self, data):
        int2char = {0: "A", 1: "B", 2: "C", 3: "D"}
        data_dict = {
            "instruction": "Answer the following medical questions by using four choices: A, B, C, D.\n",
            "input": data["question"]
            + "\nChoices:\n"
            + f"A. {data['opa']}\nB. {data['opb']}\nC. {data['opc']}\nD. {data['opd']}"
            + "\nPlease choose the correct answer.",
            "output": int2char[data["cop"]],
        }
        return data_dict

    def _tokenized_prompt(self, tokenizer, data, train_on_inputs=True):
        def tokenize(prompt, add_eos_token=True):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < 512
                and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            result["labels"] = result["input_ids"].copy()
            return result

        prompter = Prompter(template_name="alpaca", verbose=False)
        full_prompt = prompter.generate_prompt(
            instruction=data["instruction"],
            input=data["input"],
            label=data["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                instruction=data["instruction"], input=data["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt
