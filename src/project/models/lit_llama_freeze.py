from lightning_template import LightningModule
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch
import numpy as np
from typing import Dict


class LitLlamaFreeze(LightningModule):
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
        self.ckpt_path = ckpt_path
        self.r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.tokenizer_path = "/data/terencewang/llama2-hf"
        self.save_path = "work_dirs/lit_llama_freeze"
        self.automatic_optimization = False
        self.freeze_ratio = 0.6
        self.freeze_idx: Dict[str, torch.Tensor] = {}

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
        self.model.config.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)
        # super().configure_model()

    def training_step(self, batch, batch_idx, dataloader_idx=None, *args, **kwargs):
        if self.current_epoch == 0:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            self.manual_backward(loss)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    index = (
                        torch.flatten(abs(param.grad))
                        .topk(int(self.freeze_ratio * param.numel()), largest=False)
                        .indices.cpu()
                        .numpy()
                    )
                    idx = torch.tensor(np.stack(np.unravel_index(index, param.shape)))
                    modified_name = name.replace(".", "_")
                    self.freeze_idx[modified_name] = idx
        else:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            optimizer = self.optimizers()
            optimizer.zero_grad()
            loss = outputs.loss
            self.manual_backward(loss)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    modified_name = name.replace(".", "_")
                    param.grad[
                        self.freeze_idx[modified_name][0],
                        self.freeze_idx[modified_name][1],
                    ] = 0
            optimizer.step()

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
