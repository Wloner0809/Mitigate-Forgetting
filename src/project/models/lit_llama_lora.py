from lightning_template import LightningModule
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from torchmetrics import ConfusionMatrix
import os
import re


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
            super().configure_model()

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
        self.predict_result = []
        self.truth = []
        self.inference_path = "work_dirs/lit_llama_lora_inference"

    def configure_model(self):
        if self.model_not_configured:
            self.model = AutoModelForCausalLM.from_pretrained(
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
            self.model.config.pad_token_id = self.model.config.eos_token_id
            super().configure_model()

    def forward(self, batch, *args, **kwargs):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {
            "loss_dict": {
                "loss_cls": outputs.loss,
                "loss_parameter": sum(p.pow(2).mean() for p in self.parameters()),
            },
            "metric_dict": {},
        }

    def on_validation_epoch_end(self, *args, **kwargs):
        super().on_validation_epoch_end(*args, **kwargs)
        self.model.save_pretrained(self.save_path)  # transformers save model

    def predict_forward(self, batch, *args, **kwargs):
        super().predict_forward(*args, **kwargs)
        generation_cfg = {
            "max_new_tokens": 3,
            "num_return_sequences": 1,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.75,
            "top_k": 40,
            "num_beams": 4,
            "repetition_penalty": 1.05,
        }
        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **generation_cfg,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        generated_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        truth = batch["truth"]
        for i, sentence in enumerate(generated_sentence):
            predicted = sentence.split("### Response:")[1].strip()
            match = re.search(r"[A-D]", predicted)
            if match:
                predicted = match.group(0)
            else:
                predicted = "None"
            self.predict_result.append(predicted)
            self.truth.append(truth[i])

    def on_predict_end(self) -> None:
        super().on_predict_end()
        with open(os.path.join(self.inference_path, "predict_result.txt"), "w") as f:
            for sentence in self.predict_result:
                f.write(sentence + "\n")
        with open(os.path.join(self.inference_path, "truth.txt"), "w") as f:
            for answer in self.truth:
                f.write(answer + "\n")
        total = len(self.predict_result)
        acc = 0
        for i in range(len(self.predict_result)):
            match = re.search(r"[A-D]", self.truth[i])
            if match:
                truth = match.group(0)
            predicted = self.predict_result[i]
            if truth == predicted:
                acc += 1
        with open(os.path.join(self.inference_path, "accuracy.txt"), "w") as f:
            f.write(f"Accuracy: {acc/total}")
