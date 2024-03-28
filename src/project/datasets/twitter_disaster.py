from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets


class TwitterDataset(Dataset):
    # split the dataset into train, valid, test with ratio 8:1:1
    SAMPLE_NUM = {"train": 8700, "valid": 1088, "test": 1088}

    def __init__(self, subset) -> None:
        super().__init__()
        self.subset = subset
        self.data_path = "/data/terencewang/twitter_disaster"
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
        datasets = [dataset["train"], dataset["test"]]
        datasets = concatenate_datasets(datasets)
        datasets = datasets.remove_columns(["id", "keyword", "location"])

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_datasets = datasets.map(
            lambda x: tokenizer(
                x["text"], padding="max_length", truncation=True, max_length=256
            ),
            batched=True,
        )
        tokenized_datasets = tokenized_datasets.rename_column("target", "labels")
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return tokenized_datasets
