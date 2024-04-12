from lightning_template import LightningModule
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch
import numpy as np
from typing import Dict, List


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
        self.gradient_path = "work_dirs/lit_llama_gradient/lit_llama_gradient.txt"
        self.param_path = "work_dirs/lit_llama_gradient/lit_llama_param.txt"
        self.automatic_optimization = False
        self.freeze_ratio = 0.6
        self.freeze_idx: Dict[str, torch.Tensor] = {}
        self.freeze_name: List[str] = []

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
            if self.trainer.is_last_batch:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        l1_norm = torch.norm(param.grad, p=1)
                        modified_name = name.replace(".", "_")
                        self.freeze_idx[modified_name] = l1_norm
                with open(self.gradient_path, "w") as f:
                    f.write(str(self.freeze_idx) + "\n")
        else:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            sorted_idx = sorted(self.freeze_idx.items(), key=lambda x: x[1])
            freeze_num = int(len(sorted_idx) * self.freeze_ratio)
            for i in range(freeze_num):
                name = sorted_idx[i][0]
                self.freeze_name.append(name)

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    modified_name = name.replace(".", "_")
                    if modified_name in self.freeze_name:
                        param.requires_grad = False

            optimizer = self.optimizers()
            optimizer.zero_grad()
            loss = outputs.loss
            self.manual_backward(loss)
            optimizer.step()

    def on_train_epoch_end(self, *args, **kwargs):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.print_trainable_parameters()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with open(self.param_path, "a") as f:
                    f.write(name + "\n")
                    f.write(str(param) + "\n")

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


class LitLlamaFreeze_Baseline(LightningModule):
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
            if self.trainer.is_last_batch:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        index = (
                            torch.flatten(abs(param.grad))
                            .topk(int(self.freeze_ratio * param.numel()), largest=False)
                            .indices.cpu()
                            .numpy()
                        )
                        idx = torch.tensor(
                            np.stack(np.unravel_index(index, param.shape))
                        )
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

    def on_train_epoch_end(self, *args, **kwargs):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()

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
