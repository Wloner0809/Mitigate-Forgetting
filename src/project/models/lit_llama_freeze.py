from lightning_template import LightningModule
from transformers import LlamaForCausalLM
import torch
import numpy as np
from typing import Dict, List

# import wandb
import matplotlib.pyplot as plt
import os


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
        self.tokenizer_path = ckpt_path
        # self.save_path = "work_dirs/lit_llama_freeze"
        self.save_path = "/data1/terencewang/lit_llama_freeze"
        self.gradient_norm_path = (
            "work_dirs/lit_llama_gradient/lit_llama_gradient_norm.txt"
        )
        # self.pic_path = "work_dirs/lit_llama_gradient/"
        self.automatic_optimization = False
        self.freeze_ratio = 0.96
        self.freeze_layer_num = 31
        self.freeze_idx: Dict[str, torch.Tensor] = {}
        self.grad_save: Dict[str, torch.Tensor] = {}
        self.grad_before_backward: Dict[str, torch.Tensor] = {}
        self.grad_after_backward: Dict[str, torch.Tensor] = {}
        self.freeze_name: List[str] = []

    def configure_model(self):
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.ckpt_path,
        )
        # freeze some layers due to memory limitation
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
        for name, param in self.model.model.layers[
            self.freeze_layer_num :
        ].named_parameters():
            if "mlp" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
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

            # # prevent gradient explosion if needed
            # def hook(grad):
            #     l2norm = torch.norm(grad, p=2)
            #     maxnorm = 50000
            #     if l2norm > maxnorm:
            #         return grad * (maxnorm / l2norm)
            #     else:
            #         return grad

            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         param.register_hook(hook)

            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad = param.grad * batch_idx / (batch_idx + 1)

            if batch_idx % 64 == 1:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        modified_name = name.replace(".", "_")
                        self.grad_before_backward[modified_name] = param.grad
                self.model.zero_grad()

            loss = outputs.loss
            self.manual_backward(loss / (batch_idx + 1))

            if batch_idx % 64 == 1:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        modified_name = name.replace(".", "_")
                        self.grad_after_backward[modified_name] = param.grad
                tensor_before = torch.cat(
                    [abs(v.view(-1)) for v in self.grad_before_backward.values()]
                )
                tensor_after = torch.cat(
                    [abs(v.view(-1)) for v in self.grad_after_backward.values()]
                )
                self.log(
                    "error_ratio",
                    sum((tensor_before / 128) > tensor_after).item()
                    / tensor_before.size(0),
                )
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        modified_name = name.replace(".", "_")
                        param.grad = (
                            self.grad_before_backward[modified_name]
                            + self.grad_after_backward[modified_name]
                        )
                del tensor_before, tensor_after
                torch.cuda.empty_cache()

            if self.trainer.is_last_batch:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        l1_norm = torch.norm(param.grad, p=1)
                        modified_name = name.replace(".", "_")
                        self.freeze_idx[modified_name] = l1_norm
                with open(self.gradient_norm_path, "w") as f:
                    f.write(str(self.freeze_idx))
            self.log("loss_freeze", outputs.loss)
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
            self.log("loss_freeze", outputs.loss)

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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        # record log gradient and plot histogram
        if self.current_epoch == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    modified_name = name.replace(".", "_")
                    self.grad_save[modified_name] = (
                        torch.log2(abs(param.grad)).view(-1)
                        # abs(param.grad).view(-1).tolist()
                    )
            tensor_all = torch.cat([v for v in self.grad_save.values()])
            list_all = tensor_all.tolist()
            # wandb.log(
            #     {
            #         f"grad_{modified_name}": wandb.Histogram(
            #             np_histogram=np.histogram(
            #                 self.grad_save[modified_name],
            #             )
            #         )
            #     }
            # )
            if batch_idx % 64 == 0:
                plt.hist(
                    list_all,
                    bins=100,
                    range=(-65, 35),
                    color="#f9766e",
                    alpha=0.8,
                )
                plt.title(
                    f"{modified_name}",
                    loc="center",
                    fontweight="bold",
                )
                plt.xlabel("abs(gradient)", loc="center", fontweight="bold")
                plt.ylabel("Frequency", loc="center", fontweight="bold")
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.gca().spines["left"].set_linestyle("-")
                plt.gca().spines["left"].set_linewidth(2.5)
                plt.gca().spines["bottom"].set_linestyle("-")
                plt.gca().spines["bottom"].set_linewidth(2.5)
                plt.tight_layout()
                plt.savefig(os.path.join(self.pic_path, f"{batch_idx}.png"))
                plt.cla()
                plt.clf()
                plt.close("all")

    def on_validation_epoch_end(self, *args, **kwargs):
        super().on_validation_epoch_end(*args, **kwargs)
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
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
        self.tokenizer_path = ckpt_path
        self.save_path = "work_dirs/lit_llama_freeze"
        self.automatic_optimization = False
        self.freeze_ratio = 0.98
        self.freeze_idx: Dict[str, torch.Tensor] = {}

    def configure_model(self):
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.ckpt_path,
        )
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
            self.log("loss_freeze", outputs.loss)
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
            self.log("loss_freeze", outputs.loss)

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
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.save_pretrained(self.save_path)
