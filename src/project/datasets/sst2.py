from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets


class Sst2Dataset(Dataset):
    SAMPLE_NUM = {"train": 67349, "valid": 872, "test": 1821}

    def __init__(self, subset) -> None:
        super().__init__()
        self.subset = subset
        self.data_path = "/data/terencewang/sst2/data"
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
        datasets = [dataset["train"], dataset["validation"], dataset["test"]]
        datasets = concatenate_datasets(datasets)
        datasets = datasets.remove_columns(["idx"])

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_datasets = datasets.map(
            lambda x: tokenizer(
                x["sentence"], padding="max_length", truncation=True, max_length=512
            ),
            batched=True,
        )
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
        tokenized_datasets.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return tokenized_datasets
