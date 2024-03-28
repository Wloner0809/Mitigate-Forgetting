from lightning_template import LightningModule
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from torchmetrics import ConfusionMatrix
import os


class LitLlamaLora(LightningModule):
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
        self.r = 16
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
