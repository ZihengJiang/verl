import torch
from transformers import AutoConfig, AutoModelForCausalLM
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch import nn, optim
from torch.optim import AdamW
from ..base import Engine
from .config import FSDPEngineConfig
from verl.utils.import_utils import import_external_libs
import verl.utils.torch_functional as verl_F
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from contextlib import nullcontext


default_engine_config = FSDPEngineConfig()


class FSDPEngine(Engine):
    def __init__(self, config: FSDPEngineConfig):
        self.config = config
        import_external_libs(self.config.model.external_lib)
        self.use_ce_loss_fusion = self.config.model.use_ce_loss_fusion
        # self.compute_entropy_loss = torch.compile(core_algos.compute_entropy_loss, dynamic=True)
        # self.entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        # NOTE: loss related could set in `driver code`, only set default here
        # self.loss_fn = default_pg_loss_fn
        # self.loss_config = default_loss_config
        assert config.model.use_rmpad is True
        assert config.model.lora_rank == 0
        assert config.system.ulysses_sequence_parallel_size == 1
        self._build_mesh(self.config.system)


    def _build_mesh(self, system_config):
        world_size = dist.get_world_size()
        self.device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        dp_size = world_size // system_config.ulysses_sequence_parallel_size
        self.ulysses_device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(dp_size, system_config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))


    def init_model_and_optimizer(self):
        model_config = self.config.model
        system_config = self.config.system
        optim_config = self.config.optim

        self._build_model(model_config, system_config)
        self._build_optimizer(optim_config)

    def _build_model(self, model_config, system_config):    
        local_model_path = copy_to_local(src=model_config.path, verbose=True)
        if model_config.external_lib is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(model_config.external_lib)

        # load config first
        hf_config = AutoConfig.from_pretrained(local_model_path)
        init_context = get_init_weight_context_manager(use_meta_tensor=not hf_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context():
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=hf_config,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2"
            )

        if model_config.use_rmpad or system_config.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(model=self.model, ulysses_sp_size=system_config.ulysses_sequence_parallel_size)

        if model_config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        cpu_offload = None
        if system_config.param_offload:
            cpu_offload = CPUOffload(offload_params=system_config.param_offload)

        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )


    def _build_optimizer(self, optim_config):
        assert optim_config.type == "adam"
        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=optim_config.lr,
            betas=optim_config.betas,
            weight_decay=optim_config.weight_decay,
        )
        

    def forward_backward_step(self, batch):
        self.model.train()

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        position_ids = batch["position_ids"].cuda()
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).cuda()
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = nullcontext()
        with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Standard forward pass without sequence parallel
            labels = input_ids[:, 1:].contiguous()
            outputs = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels.contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_labels)
            loss = loss * loss_mask.to(loss.device)

        valid_token_this_rank = torch.sum(loss_mask)
        dp_size = 1

        loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size
        outputs.loss = loss
        loss.backward()

        return outputs
        raise ValueError
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