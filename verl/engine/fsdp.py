from .base import Engine
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW



@dataclass
class FSDPConfig:
    model_path: str
    optim: str = "adam"


class FSDPEngine(Engine):
    def __init__(self, config: FSDPConfig):
        super().__init__(config)
        self.config = config

    def init_model_and_optimizer(self):
        self._build_model(self.config)
        self._build_optimizer(self.config)


    def _build_model(self, config):
        rank = dist.get_rank()
        device_map={"": rank}
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model = FSDP(self.model)


    def _build_optimizer(self, config):
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        

    def forward_backward_step(self, batch):
        self.model.train()
        rank = dist.get_rank()
        input_ids = batch['input_ids'].to(rank)
        attention_mask = batch['attention_mask'].to(rank)
        labels = batch['labels'].to(rank)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute loss
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        outputs.loss = loss
        loss.backward()
        return outputs


    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()


    def optimizer_step(self):
        self.optimizer.step()
    
    def lr_scheduler_step(self):
        pass

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn