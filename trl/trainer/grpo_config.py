from dataclasses import dataclass, field
from typing import Optional, List, Union

from transformers import TrainingArguments

from trl.trainer.reward_config import RewardConfig


@dataclass
class GRPOConfig(TrainingArguments):
    """
    GRPOConfig collects all training arguments related to the [`GRPOTrainer`] class.

    Args:
        beta (`float`, defaults to 0.1):
            The beta parameter for GRPO. Higher values lead to more regularization, which can help prevent catastrophic
            forgetting but may slow down learning.
        reference_learning_rate (`float`, defaults to None):
            The learning rate for the reference model. If None, the learning rate for the policy will be used.
        group_by_length (`bool`, defaults to False):
            Whether to group samples by length (in tokens) when creating batches. This can lead to more efficient
            training, especially with mixed modalities, by grouping samples with similar lengths together.
        max_prompt_length (`int`, defaults to 1024):
            Maximum length of the prompt to be used. Prompts longer than this will be truncated.
        max_completion_length (`int`, defaults to 1024):
            Maximum length of the completion to be used. Completions longer than this will be truncated.
        reward_adapter_path (`str`, defaults to None):
            Path to the adapter weights to be loaded for the reward model.
        load_in_8bit (`bool`, defaults to False):
            Whether to load the reward model in 8-bit precision.
        load_in_4bit (`bool`, defaults to False):
            Whether to load the reward model in 4-bit precision.
        reward_adapter_name_or_path (`str`, defaults to None):
            Name or path of the reward adapter to be loaded.
        torch_dtype (`Union[str, torch.dtype]`, defaults to None):
            Datatype to use for training. If provided as a string, it must be one of "float32", "float16", "bfloat16".
    """

    beta: float = field(default=0.1, metadata={"help": "The beta parameter for GRPO."})
    reference_learning_rate: Optional[float] = field(
        default=None, metadata={"help": "The learning rate for the reference model. If None, defaults to learning_rate."}
    )
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to group samples of roughly the same length together when batching. "
            "If True, longer sequences will generally be processed first within each batch."
        },
    )
    max_prompt_length: int = field(default=1024, metadata={"help": "Maximum length of the prompt to be used."})
    max_completion_length: int = field(default=1024, metadata={"help": "Maximum length of the completion to be used."})
    reward_adapter_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the adapter weights to be loaded for the reward model."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the reward model in 8-bit precision."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the reward model in 4-bit precision."})
    reward_adapter_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Name or path of the reward adapter to be loaded."}
    )
    torch_dtype: Optional[Union[str, "torch.dtype"]] = field(
        default=None,
        metadata={
            "help": "Datatype to use for training. If provided as a string, it must be one of 'float32', 'float16', 'bfloat16'."
        },
    )
    reward_configs: List[RewardConfig] = field(
        default_factory=list,
        metadata={"help": "List of reward configs to be used by SLiC-HF trainer."},
    )
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    ) 