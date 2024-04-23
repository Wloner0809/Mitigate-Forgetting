from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import LlamaTokenizer
import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = osp.join("src/project/datasets/", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class MedmcqaDataset(Dataset):
    SAMPLE_NUM = {"train": 5000, "valid": 1000, "test": 1000}

    def __init__(self, subset) -> None:
        super().__init__()
        self.subset = subset
        # self.data_path = "/data/terencewang/medmcqa/data"
        self.data_path = "/home/wf/Projects/wangyu/data/medmcqa"
        # self.tokenizer_path = "/data/terencewang/llama2-hf"
        self.tokenizer_path = "/home/wf/Projects/wangyu/model/llama2-hf"
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

        if self.subset == "train":
            all_data = train_data.select(range(self.SAMPLE_NUM["train"]))
        elif self.subset == "valid":
            all_data = train_data.select(
                range(
                    self.SAMPLE_NUM["train"],
                    self.SAMPLE_NUM["train"] + self.SAMPLE_NUM["valid"],
                )
            )
        elif self.subset == "test":
            all_data = train_data.select(
                range(
                    self.SAMPLE_NUM["train"] + self.SAMPLE_NUM["valid"],
                    self.SAMPLE_NUM["train"]
                    + self.SAMPLE_NUM["valid"]
                    + self.SAMPLE_NUM["test"],
                )
            )

        # set tokenizer as follows
        tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        tokenizer.pad_token = "<unk>"
        tokenizer.padding_side = "left"

        all_data = all_data.map(
            lambda x: self._tokenized_prompt(tokenizer, x),
            remove_columns=["instruction", "output"],
        )
        all_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return all_data

    def _preprocess(self, data):
        int2char = {0: "A", 1: "B", 2: "C", 3: "D"}
        data_dict = {
            "instruction": "Solve the following medical problem by choosing the correct answer from following four choices.\nQuestion:\n"
            + data["question"]
            + "\nChoices: \n"
            + f"A.{data['opa']}, B.{data['opb']}, C. {data['opc']}, D. {data['opd']}\n Answer:",
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
            label=data["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(instruction=data["instruction"])
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt
