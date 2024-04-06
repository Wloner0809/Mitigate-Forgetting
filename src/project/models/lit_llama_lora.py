from lightning_template import LightningModule
from transformers import (
    AutoModelForSequenceClassification,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from torchmetrics import ConfusionMatrix
import os
import re


class LitLlamaLora_CausalTask(LightningModule):
    def __init__(
        self,
        ckpt_path,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.ckpt_path = ckpt_path
        self.tokenizer_path = "/data/terencewang/llama2-hf"
        self.save_path = "work_dirs/lit_llama_lora_causal"
        # self.lora_path = "work_dirs/llama_causal"
        self.inference_path = "work_dirs/lit_llama_lora_inference"
        self.predict_result = []

    def configure_model(self):
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.ckpt_path,
        )
        config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, config)
        "use in inference stage"
        # self.model = PeftModel.from_pretrained(self.model, self.lora_path)
        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)
        # super().configure_model()

    def forward(self, batch, *args, **kwargs):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {
            "loss_dict": {
                "loss_causal": outputs.loss,
                "loss_parameter": sum(p.pow(2).mean() for p in self.parameters()),
            },
            "metric_dict": {},
        }

    def on_validation_epoch_end(self, *args, **kwargs):
        super().on_validation_epoch_end(*args, **kwargs)
        # transformers save model
        self.model.save_pretrained(self.save_path)

    def predict_forward(self, batch, *args, **kwargs):
        super().predict_forward(*args, **kwargs)
        generation_cfg = {
            "max_new_tokens": 128,
            "num_return_sequences": 1,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": 0.75,
            "repetition_penalty": 1.15,
        }
        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **generation_cfg,
        )

        # set tokenizer as follows
        tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        tokenizer.pad_token = "<unk>"
        tokenizer.padding_side = "left"

        generated_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, sentence in enumerate(generated_sentence):
            print(sentence.split("### Response:")[1].strip())
            print("-" * 70)
            predicted = sentence.split("### Response:")[1].strip()
            match = re.search(r"[A-D]", predicted)
            if match:
                predicted = match.group(0)
            else:
                predicted = "None"
            self.predict_result.append(predicted)

    def on_predict_end(self) -> None:
        super().on_predict_end()
        with open(os.path.join(self.inference_path, "predict_lora.txt"), "w") as f:
            for sentence in self.predict_result:
                f.write(sentence + "\n")


class LitLlamaLora_BinaryTask(LightningModule):
    def __init__(
        self,
        ckpt_path,
        predict_tasks=None,
        *args,
        **kwargs,
    ) -> None:
        if predict_tasks is None:
            predict_tasks = ["confusion_matrix"]
        super().__init__(
            predict_tasks=predict_tasks,
            *args,
            **kwargs,
        )
        self.r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.ckpt_path = ckpt_path

    def configure_model(self) -> None:
        if self.model_not_configured:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.ckpt_path,
                num_labels=2,
            )
            config = LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=[
                    "q_proj",
                    "v_proj",
                ],
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            self.model = get_peft_model(self.model, config)
            self.model.config.pad_token_id = self.model.config.eos_token_id

            if "confusion_matrix" in self.predict_tasks:
                self.confusion = ConfusionMatrix(task="binary", num_classes=2)
            # super().configure_model()

    def forward(self, batch, *args, **kwargs):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        preds = outputs.logits.argmax(dim=1).squeeze().float()
        return {
            "loss_dict": {
                "loss_cls": outputs.loss,
                "loss_parameter": sum(p.pow(2).mean() for p in self.parameters()),
            },
            "metric_dict": {"preds": preds, "target": batch["labels"]},
        }

    def predict_forward(self, batch, *args, **kwargs):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        preds = outputs.logits.argmax(dim=1).squeeze().float()
        return {"preds": preds}

    def predict_confusion_matrix(self, batch, *args, output_path, preds, **kwargs):
        self.confusion.update(preds, batch["labels"])
        fig, _ = self.confusion.plot()
        fig.savefig(os.join(output_path, f"{batch['index'][0]}.png"))

    def on_predict_end(self) -> None:
        super().on_predict_end()
        if "confusion_matrix" in self.predict_tasks:
            fig, _ = self.confusion.plot()
            fig.savefig(
                os.path.join(self.predict_path, "confusion_matrix/confusion_matrix.png")
            )
