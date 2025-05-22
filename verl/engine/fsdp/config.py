from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HFModelConfig:
    # model config
    path: str = ""
    external_lib: str = None
    # model optimization config
    use_rmpad: bool = True
    lora_rank: int = 0
    enable_gradient_checkpointing: bool = False
    use_ce_loss_fusion: bool = False
    logits_clamp: float = 0.0
    update_gate_ema: bool = False


@dataclass
class HFOptimConfig:
    grad_clip: float = 1.0
    type: str = "adam"
    lr: float = 1e-5
    betas: List = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-08
    weight_decay: float = 0.1
    lr_warmup_steps: int = -1
    lr_warmup_steps_ratio: float = 0.0  # the total steps will be injected during runtime
    min_lr_ratio: float = 0.0  # only useful for warmup with cosine
    warmup_style: str = "constant"  # select from constant/cosine
    total_training_steps: int = -1  # must be override by program
    force_bfloat16_state: bool = False  # wether to compress optimizer states to bf16


@dataclass
class MixedPrecisionConfig:
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    buffer_dtype: str = "fp32"


@dataclass
class WrapPolicyConfig:
    transformer_layer_cls_to_wrap: List = None
    min_num_params: int = 0
    disable: bool = False


@dataclass
class FSDPSystemConfig:
    # mp
    strategy: str = (
        "fsdp"  # we need this to choice framework between [fsdp, vescale, fsdp2], bacause they use same code
    )
    mixed_precision: MixedPrecisionConfig = None
    model_dtype: str = (
        "fp32"  # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
    )
    fsdp_size: int = -1
    # act offload
    act_offload: bool = False
    act_offload_upbound: int = None
    act_offload_buff_size: int = 40
    act_offload_threshold: int = 1048576
    # param & optim offload
    param_offload: bool = False
    optim_offload: bool = False
    # sp
    ulysses_sequence_parallel_size: int = 1
    # tp
    tp_size: int = 1
    tp_outside: bool = False
    # wrap policy
    wrap_policy: WrapPolicyConfig = field(default_factory=WrapPolicyConfig)



@dataclass
class FSDPEngineConfig:
    model: HFModelConfig = field(default_factory=HFModelConfig)
    system: FSDPSystemConfig = field(default_factory=FSDPSystemConfig)
    optim: HFOptimConfig = field(default_factory=HFOptimConfig)
    micro_batch_size: int = -1  # max batch size in a microbatch
    micro_batch_tokens: int = -1  # max tokens number in a microbatch