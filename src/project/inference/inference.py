import os
import logging
from accelerate import Accelerator
import fire
import time
import torch
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from tqdm import tqdm
import json
import os.path as osp
from typing import Union
from torch.utils.data import DataLoader
from datasets import load_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


def main(
    model_path: str = "/data/terencewang/llama2-hf",
    output_dir: str = "work_dirs/lit_llama_lora_inference",
    lora_dir: str = "work_dirs/lit_llama_lora_causal",
    dataset_path: str = "/data/terencewang/medmcqa_json",
    dateset_name: str = "medmcqa",
    number_of_samples: int = 500,
):
    def get_configs():
        generation_cfg = {
            "max_new_tokens": 128,
            "temperature": 0.1,
            "top_p": 0.75,
            "repetition_penalty": 1.15,
            "do_sample": False,
        }
        inference_cfg = {
            "max_length": 1024,
            "token_batch_size": 16,
            "inference_batch_size": 12,
        }
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        prompt_config = Prompter("alpaca")
        return generation_cfg, inference_cfg, bnb_config, prompt_config

    def medmcqa_preprocess(data):
        int2char = {1: "A", 2: "B", 3: "C", 4: "D"}
        data_dict = {
            "instruction": "Solve the following medical problem by choosing the correct answer from following four choices.\nQuestion:\n"
            + data["question"]
            + "\nChoices: \n"
            + f"A.{data['opa']}, B.{data['opb']}, C.{data['opc']}, D.{data['opd']}\n Answer:",
            "answer": f"{int2char[data['cop']]}",
        }
        return data_dict

    def create_test_dataset(dataset_path: str, number_of_samples: int = 500):
        num_samples = number_of_samples
        medmcqa_path = dataset_path
        data = load_dataset(
            "json",
            data_files=os.path.join(medmcqa_path, "dev.json"),
            split="train",
        ).filter(lambda x: x["choice_type"] == "single")
        len_of_data = len(data)
        select_range = range(len_of_data - min(num_samples, len_of_data), len_of_data)
        data = data.select(select_range)
        data = (
            data.map(medmcqa_preprocess)
            .remove_columns(
                [
                    "question",
                    "exp",
                    "cop",
                    "opa",
                    "opb",
                    "opc",
                    "opd",
                    "subject_name",
                    "topic_name",
                    "id",
                    "choice_type",
                ]
            )
            .shuffle()
        )
        return data

    def get_model(model_path, lora_dir, bnb_config, accelerator):
        with accelerator.main_process_first():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=bnb_config,
            )
            if lora_dir != "":
                model = PeftModel.from_pretrained(model, lora_dir)
            else:
                logger.info("No lora directory provided, use base model")
            model.resize_token_embeddings(model.config.vocab_size + 1)
            model.config.pad_token_id = 0
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
            return model

    def get_tokenizer(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer

    def get_dataloader(dataset, tokenizer, inference_cfg, prompt_config, accelerator):
        with accelerator.main_process_first():
            dataset = dataset.map(
                lambda x: {
                    "full_prompt": prompt_config.generate_prompt(x["instruction"], None)
                }
            )
            columns = dataset.column_names
            tokenized = dataset.map(
                lambda x: tokenizer(
                    x["full_prompt"],
                    truncation=True,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=inference_cfg["max_length"],
                ),
                batched=True,
                batch_size=inference_cfg["token_batch_size"],
            )
            tokenized = tokenized.remove_columns(columns)
            data_collator = DataCollatorWithPadding(tokenizer)
            dataloader = DataLoader(
                tokenized,
                batch_size=inference_cfg["inference_batch_size"],
                collate_fn=data_collator,
            )
            return dataloader

    def run_generation(
        generation_cfg,
        prompt_config,
        dataloader,
        tokenizer,
        model,
        accelerator,
    ):
        model, dataloader = accelerator.prepare(model, dataloader)
        accelerator.wait_for_everyone()
        output_sequences = []
        start_time = time.time()
        for batch in tqdm(dataloader):
            unwrapped_model = accelerator.unwrap_model(model)
            with torch.inference_mode():
                generated_tokens = unwrapped_model.generate(**batch, **generation_cfg)
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                generated_tokens = (
                    accelerator.gather_for_metrics(generated_tokens).cpu().tolist()
                )
            generated_sentence = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            outputs = [
                prompt_config.get_response(output) for output in generated_sentence
            ]
            tqdm.write(f"Output:\n{outputs}")
            output_sequences.extend(outputs)
        end_time = time.time()
        logger.info(f"Generation time: {end_time - start_time} sec")
        return output_sequences

    if not os.path.exists(lora_dir):
        logger.error("Could not find adapter checkpoint directory")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generation_cfg, inference_cfg, bnb_config, prompt_config = get_configs()
    accelerator = Accelerator()
    logger.info("Loading tokenizer")
    tokenizer = get_tokenizer(model_path)
    logger.info("Loading datasets")
    test_dataset = create_test_dataset(dataset_path, number_of_samples)
    logger.info("Loading model")
    model = get_model(model_path, lora_dir, bnb_config, accelerator)
    logger.info("starting inference")
    dataloader = get_dataloader(
        test_dataset, tokenizer, inference_cfg, prompt_config, accelerator
    )
    output_sequences = run_generation(
        generation_cfg,
        prompt_config,
        dataloader,
        tokenizer,
        model,
        accelerator,
    )
    if accelerator.is_local_main_process:
        logger.info("Saving results")
        instruction = [p["instruction"] for p in test_dataset]
        answer = [p["answer"] for p in test_dataset]
        output = output_sequences
        with open(
            os.path.join(output_dir, f"{dateset_name}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {"instruction": instruction, "answer": answer, "output": output},
                f,
                ensure_ascii=False,
                indent=4,
            )


if __name__ == "__main__":
    fire.Fire(main)
