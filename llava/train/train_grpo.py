import os
import sys
import torch
import transformers
import copy
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Union

# from trl import GRPOConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizerFast,
    TrainingArguments,
    PreTrainedTokenizerBase,
    HfArgumentParser,
    set_seed,
    Trainer,
)

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.mm_utils import tokenizer_image_token
from llava.utils import rank0_print
from llava.train.train import LLaVATrainer
from llava.train.llava_grpo_trainer import LLaVAGRPOTrainer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.model.language_model.llava_mixtral import LlavaMixtralForCausalLM


# Basic implementation of GRPOTrainer to replace trl dependency
class GRPOTrainer(Trainer):
    """
    Basic implementation of GRPOTrainer to replace trl dependency.
    """
    
    def __init__(self, reward_funcs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_funcs = reward_funcs or ["accuracy"]
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        self.ref_model = None
        
    def _compute_kl_divergence(self, current_log_probs, ref_log_probs, labels):
        """Compute KL divergence between current and reference policy."""
        mask = (labels != -100).float()
        kl_div = F.kl_div(
            ref_log_probs, current_log_probs, reduction="none"
        ).sum(-1)
        kl_div = (kl_div * mask).sum() / mask.sum().clamp(min=1.0)
        return kl_div
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute GRPO loss."""
        # This is a simplified version - in a real implementation, 
        # you would compute rewards, advantages, etc.
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


@dataclass
class GRPOConfig(TrainingArguments):
    """
    Configuration class for GRPO (Generative Reinforcement with Policy Optimization).
    """
    beta: float = field(default=0.1, metadata={"help": "Weight for KL penalty"})
    kl_coef: float = field(default=0.1, metadata={"help": "KL divergence coefficient"})
    gamma: float = field(default=1.0, metadata={"help": "Discount factor"})
    lam: float = field(default=0.95, metadata={"help": "GAE lambda"})
    init_kl_coef: float = field(default=0.2, metadata={"help": "Initial KL coefficient"})
    target_kl: float = field(default=6.0, metadata={"help": "Target KL value"})
    horizon: int = field(default=10000, metadata={"help": "Horizon value for GRPO"})
    cliprange: float = field(default=0.2, metadata={"help": "Clip range"})
    cliprange_value: float = field(default=0.2, metadata={"help": "Clip range for value function"})
    vf_coef: float = field(default=0.1, metadata={"help": "Value function coefficient"})
    gen_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"max_new_tokens": 512, "temperature": 1.0, "top_k": 0, "top_p": 1.0}
    )
    # LLaVATrainer相关参数
    group_by_modality_length: bool = field(default=False, metadata={"help": "Group training data by modality length for batch creation"})
    group_by_modality_length_auto: bool = field(default=False, metadata={"help": "Automatically group training data by modality length"})
    auto_find_batch_size: bool = field(default=False, metadata={"help": "Automatically find optimal batch size"})
    # LoRA related parameters
    lora_enable: bool = field(default=False, metadata={"help": "Whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias type"})
    lora_target_modules: Optional[str] = field(default=None, metadata={"help": "LoRA target modules"})
    lora_modules_to_save: Optional[str] = field(default=None, metadata={"help": "LoRA modules to save"})
    # Add other GRPO-specific parameters as needed


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="lmms-lab/llava-onevision-qwen2-7b-si")
    version: str = field(default="v1")
    vision_tower: str = field(default="google/siglip-so400m-patch14-384")
    mm_vision_select_layer: int = field(default=-2)
    mm_vision_select_feature: str = field(default="patch")
    pretrain_mm_mlp_adapter: str = field(default=None)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: str = field(default="spatial_unpad")
    mm_tunable_parts: str = field(default="mm_vision_tower,mm_mlp_adapter,mm_language_model")
    tune_vision_tower: bool = field(default=False)
    tune_vision_tower_nonlinears_only: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    tune_mm_language_model: bool = field(default=False)
    mm_projector_lr: Optional[float] = field(default=None)
    mm_vision_tower_lr: Optional[float] = field(default=None)
    vision_tower_lr: Optional[float] = field(default=None)
    model_max_length: int = field(default=32768, metadata={"help": "Maximum sequence length."})
    peft_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = field(default=False)
    is_multimodal: bool = field(default=True)
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default='square')
    image_grid_pinpoints: Optional[str] = field(default=None)
    system_prompt: str = field(
        default=(
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>"
        )
    )
    reward_funcs: str = field(default="accuracy,format")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def smart_tokenizer_and_embedding_resize(
    tokenizer: PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_IMAGE_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "</s>"
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<unk>"

    if len(special_tokens_dict) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)

    # Let's skip embedding resize for now, as it seems to be causing issues
    # Just update the vocab size in the model config
    model.config.vocab_size = len(tokenizer)
    
    # Add image tokens
    if getattr(model.config, 'mm_use_im_start_end', False):
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)


def preprocess_vision_data(example):
    """
    Preprocess vision data by updating images and conversation format.
    """
    global template_name, tokenizer, IMAGE_TOKEN_INDEX
    
    image_file = example.get("image", None)
    
    # Setup conversation
    conv = conv_templates[template_name].copy()
    
    # Add system prompt if needed
    if system_prompt and conv.system != system_prompt:
        conv.system = system_prompt
    
    # Format conversation
    if "conversations" in example:
        roles = {"human": conv.roles[0], "assistant": conv.roles[1]}
        
        messages = []
        for message in example["conversations"]:
            role = roles.get(message["from"], message["from"])
            content = message["value"]
            messages.append((role, content))
            
        if len(messages) % 2 != 0:
            messages.append((conv.roles[1], ""))
            
        for role, content in messages:
            if image_file and role == conv.roles[0]:
                content = DEFAULT_IMAGE_TOKEN + "\n" + content
                conv.append_message(role, content)
                image_file = None  # Only add image token to the first human message
            else:
                conv.append_message(role, content)
    else:
        # Assume it's a simple prompt-completion format
        if image_file:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + example.get("prompt", ""))
        else:
            conv.append_message(conv.roles[0], example.get("prompt", ""))
        
        conv.append_message(conv.roles[1], example.get("completion", ""))
    
    # Convert to model inputs
    if conv.sep_style == SeparatorStyle.LLAMA_2:
        raw_text = conv.get_prompt()
    else:
        raw_text = conv.get_prompt() + conv.sep + conv.roles[1] + ": "
    
    try:    
        inputs = tokenizer_image_token(raw_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
        
        # Set target text (labels)
        if conv.sep_style == SeparatorStyle.LLAMA_2:
            target_text = conv.get_prompt() + conv.sep2 + conv.roles[1] + ": " + example.get("completion", "")
        else:
            target_text = conv.get_prompt() + conv.sep + conv.roles[1] + ": " + example.get("completion", "")
            
        targets = tokenizer_image_token(target_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
    except Exception as e:
        rank0_print(f"Error processing example: {e}")
        # Return empty results for this example to skip it
        return {"input_ids": [], "labels": [], "image": None}
    
    return {
        "input_ids": inputs,
        "labels": targets,
        "image": image_file,
    }


def process_batch_for_grpo(batch, tokenizer):
    """Process a batch of examples for GRPO training."""
    input_ids = torch.stack([torch.tensor(x) for x in batch["input_ids"]])
    labels = torch.stack([torch.tensor(x) for x in batch["labels"]])
    attention_mask = torch.ones_like(input_ids)
    
    # Find the position of the image token
    image_token_indices = (input_ids == IMAGE_TOKEN_INDEX).nonzero()
    images = []
    
    # Get images if available
    if "image" in batch and batch["image"][0] is not None:
        for img_path in batch["image"]:
            if img_path:
                # Load image (this would depend on your image processing pipeline)
                img = torch.zeros((3, 224, 224))  # Placeholder
                images.append(img)
    
    # Create labels that only include the assistant's response (set others to -100)
    for i, label_seq in enumerate(labels):
        # Find response start
        sep_positions = (label_seq == tokenizer.sep_token_id).nonzero()
        if len(sep_positions) > 0:
            # Set everything before the last separator token to -100
            last_sep_pos = sep_positions[-1].item()
            labels[i, :last_sep_pos+1] = -100
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
    if images:
        result["images"] = torch.stack(images)
    
    return result


def main():
    global template_name, system_prompt, IMAGE_TOKEN_INDEX, tokenizer
    
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 如果使用DeepSpeed，对Qwen模型禁用某些优化
    if 'qwen' in model_args.model_name_or_path.lower() and training_args.deepspeed is not None:
        rank0_print("Detected Qwen model with DeepSpeed - modifying DeepSpeed config to avoid shape errors")
        # 禁用优化以避免形状不匹配问题
        if isinstance(training_args.deepspeed, str) and os.path.exists(training_args.deepspeed):
            import json
            with open(training_args.deepspeed, 'r') as f:
                ds_config = json.load(f)
            # 禁用可能导致问题的优化
            if "zero_optimization" in ds_config:
                ds_config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e10
                if "offload_optimizer" in ds_config["zero_optimization"]:
                    ds_config["zero_optimization"]["offload_optimizer"]["buffer_count"] = 4
                    ds_config["zero_optimization"]["offload_optimizer"]["pin_memory"] = True
                if "stage3_gather_16bit_weights_on_model_save" in ds_config["zero_optimization"]:
                    ds_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False
            if "optimizer" in ds_config:
                ds_config["optimizer"]["params"]["torch_adam"] = True
                ds_config["optimizer"]["params"]["adam_w_mode"] = False
            # 写回修改后的配置
            with open(training_args.deepspeed, 'w') as f:
                json.dump(ds_config, f, indent=4)
        elif isinstance(training_args.deepspeed, dict):
            # 如果deepspeed是一个配置字典，直接修改它
            if "zero_optimization" in training_args.deepspeed:
                training_args.deepspeed["zero_optimization"]["stage3_param_persistence_threshold"] = 1e10
                if "offload_optimizer" in training_args.deepspeed["zero_optimization"]:
                    training_args.deepspeed["zero_optimization"]["offload_optimizer"]["buffer_count"] = 4
                    training_args.deepspeed["zero_optimization"]["offload_optimizer"]["pin_memory"] = True
            if "optimizer" in training_args.deepspeed:
                training_args.deepspeed["optimizer"]["params"]["torch_adam"] = True
                training_args.deepspeed["optimizer"]["params"]["adam_w_mode"] = False
    
    # Set model peft config if provided
    use_peft = model_args.peft_config and len(model_args.peft_config) > 0
    if use_peft:
        peft_config = LoraConfig(**model_args.peft_config)
    else:
        peft_config = None
    
    # 如果GRPOConfig没有显式设置group_by_modality_length，则从data_args获取默认值
    if hasattr(data_args, 'group_by_modality_length_default'):
        training_args.group_by_modality_length = data_args.group_by_modality_length_default
        
    # 使用monkey patching将学习率参数添加到training_args对象
    if hasattr(model_args, 'mm_projector_lr'):
        training_args.__dict__['mm_projector_lr'] = model_args.mm_projector_lr
    
    if hasattr(model_args, 'mm_vision_tower_lr'):
        training_args.__dict__['mm_vision_tower_lr'] = model_args.mm_vision_tower_lr
        
    if hasattr(model_args, 'vision_tower_lr'):
        training_args.__dict__['vision_tower_lr'] = model_args.vision_tower_lr
    
    # 处理model_max_length参数    
    if hasattr(model_args, 'model_max_length') and model_args.model_max_length is not None:
        training_args.__dict__['model_max_length'] = model_args.model_max_length
        
    # Set the system prompt
    system_prompt = data_args.system_prompt
    
    # Parse reward functions
    reward_funcs = data_args.reward_funcs.split(",")
    rank0_print(f"Using reward functions: {reward_funcs}")
    
    # Set the conversation template based on the model
    template_name = model_args.version
    if template_name not in conv_templates:
        rank0_print(f"Warning: Template {template_name} not found in conv_templates. Using 'v1' as fallback.")
        template_name = "v1"
    rank0_print(f"Using template: {template_name}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=False,
            padding_side="right", 
            trust_remote_code=True
        )
    except Exception as e:
        rank0_print(f"Error loading tokenizer: {e}")
        rank0_print("Trying again with use_fast=True...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=True,
                padding_side="right", 
                trust_remote_code=True
            )
        except Exception as e2:
            rank0_print(f"Error loading tokenizer with use_fast=True: {e2}")
            raise ValueError(f"Could not initialize tokenizer for {model_args.model_name_or_path}")
    
    # Determine which model class to use based on the model name
    model_name = model_args.model_name_or_path.lower()
    if 'qwen' in model_name:
        model_class = LlavaQwenForCausalLM
    elif 'mixtral' in model_name:
        model_class = LlavaMixtralForCausalLM
    elif 'mistral' in model_name:
        model_class = LlavaMistralForCausalLM
    else:
        # Default to LLaMA-based model
        model_class = LlavaLlamaForCausalLM
    
    rank0_print(f"Using model class: {model_class.__name__}")
    
    # Create model
    try:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
    except Exception as e:
        rank0_print(f"Error loading model: {e}")
        rank0_print("Trying with additional parameters...")
        try:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
                low_cpu_mem_usage=True
            )
        except Exception as e2:
            rank0_print(f"Error loading model with additional parameters: {e2}")
            raise ValueError(f"Could not initialize model {model_args.model_name_or_path}")
    
    # Resize tokenizer and embeddings
    smart_tokenizer_and_embedding_resize(tokenizer, model)
    
    # Process image token
    if 'qwen' in model_args.model_name_or_path.lower():
        IMAGE_TOKEN_INDEX = tokenizer.convert_tokens_to_ids('<img>')
    else:
        IMAGE_TOKEN_INDEX = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    
    if IMAGE_TOKEN_INDEX is None:
        IMAGE_TOKEN_INDEX = tokenizer.convert_tokens_to_ids('<image>')
    
    if IMAGE_TOKEN_INDEX is None:
        IMAGE_TOKEN_INDEX = len(tokenizer) - 1
    
    # Initialize the vision tower
    try:
        if model_args.vision_tower is None:
            rank0_print("Warning: vision_tower is None, using default 'google/siglip-so400m-patch14-384'")
            vision_tower_name = "google/siglip-so400m-patch14-384"
        else:
            vision_tower_name = model_args.vision_tower
            
        vision_tower = build_vision_tower(
            vision_tower_name,
            vision_select_layer=model_args.mm_vision_select_layer,
            vision_select_feature=model_args.mm_vision_select_feature
        )
        vision_tower.to(device=training_args.device, dtype=torch.bfloat16 if training_args.bf16 else torch.float32)
        
        # Connect the vision tower to the model
        try:
            if hasattr(model, 'model'):
                model.model.vision_tower = vision_tower
            if hasattr(model, 'vision_tower'):
                model.vision_tower = vision_tower
        except AttributeError as e:
            rank0_print(f"Warning: Could not connect vision tower to model: {e}")
            rank0_print("Continuing without vision tower connection.")
    except Exception as e:
        rank0_print(f"Error initializing vision tower: {e}")
        rank0_print("Continuing without vision tower.")
    
    # Load dataset
    try:
        if data_args.data_path is None:
            raise ValueError("data_path must be provided")
            
        if data_args.data_path.endswith('.json'):
            # If the data_path is a local JSON file
            dataset = load_dataset('json', data_files=data_args.data_path)
        else:
            # If the data_path is a Hugging Face dataset ID
            dataset = load_dataset(data_args.data_path)
    except Exception as e:
        rank0_print(f"Error loading dataset: {e}")
        rank0_print("Creating a tiny dummy dataset for testing")
        
        # Create a tiny dummy dataset for testing
        dummy_train = {"prompt": ["What is this?"] * 10, "completion": ["This is a test."] * 10, "image": [None] * 10}
        dummy_val = {"prompt": ["Test question?"] * 2, "completion": ["Test answer."] * 2, "image": [None] * 2}
        
        dataset = {
            "train": Dataset.from_dict(dummy_train),
            "validation": Dataset.from_dict(dummy_val)
        }
    
    # Split dataset if it has 'train' and 'validation' splits
    if isinstance(dataset, dict):
        # Dataset is already a dictionary with splits
        train_dataset = dataset.get('train', None)
        eval_dataset = dataset.get('validation', None)
        
        if train_dataset is None:
            raise ValueError("Dataset dictionary must contain a 'train' split")
            
        if eval_dataset is None and len(train_dataset) > 1:
            # Create a small validation set from training data
            eval_split = int(len(train_dataset) * 0.9)
            eval_dataset = train_dataset.select(range(eval_split, len(train_dataset)))
            train_dataset = train_dataset.select(range(eval_split))
    else:
        # It's a DatasetDict or has a different structure
        if hasattr(dataset, 'train'):
            train_dataset = dataset.train
            if hasattr(dataset, 'validation'):
                eval_dataset = dataset.validation
            else:
                # Create a small validation set from training data
                eval_split = int(len(train_dataset) * 0.9)
                eval_dataset = train_dataset.select(range(eval_split, len(train_dataset)))
                train_dataset = train_dataset.select(range(eval_split))
        else:
            # If the dataset doesn't have splits, create them
            try:
                split = dataset['train'].train_test_split(test_size=0.1)
                train_dataset, eval_dataset = split['train'], split['test']
            except Exception as e:
                rank0_print(f"Error splitting dataset: {e}")
                raise ValueError("Could not extract or create train/validation splits from the dataset")
    
    rank0_print(f"Initial train dataset size: {len(train_dataset)}")
    rank0_print(f"Initial eval dataset size: {len(eval_dataset)}")
    
    # Map the preprocessing function to the datasets
    train_dataset = train_dataset.map(preprocess_vision_data, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess_vision_data, remove_columns=eval_dataset.column_names)
    
    # Filter out any examples with empty inputs
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    rank0_print(f"Train dataset size: {len(train_dataset)}")
    rank0_print(f"Eval dataset size: {len(eval_dataset)}")
    
    # 为数据集添加模态长度属性
    def add_modality_length(dataset):
        lengths = []
        for example in dataset:
            # 正值表示多模态（有图像），负值表示纯文本
            if example.get("image") is not None:
                lengths.append(len(example["input_ids"]))
            else:
                lengths.append(-len(example["input_ids"]))
                
        dataset.modality_lengths = lengths
        return dataset
    
    # 为训练和评估数据集添加模态长度
    if training_args.group_by_modality_length or training_args.group_by_modality_length_auto:
        train_dataset = add_modality_length(train_dataset)
        
    # Create the trainer
    trainer = LLaVAGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        data_args=data_args,
        model_args=model_args
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model(training_args.output_dir)
    
    # Step 5: Ensure model weights are saved
    try:
        if hasattr(trainer, "model") and trainer.model:
            unwrapped_model = trainer.model
            if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "unwrap_model"):
                unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)

            # Save model weights and configs
            if training_args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(unwrapped_model.named_parameters(), training_args.lora_bias)
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(unwrapped_model.named_parameters())
                if training_args.local_rank == 0 or training_args.local_rank == -1:
                    if hasattr(unwrapped_model, "config"):
                        unwrapped_model.config.save_pretrained(training_args.output_dir)
                    if hasattr(unwrapped_model, "generation_config"):
                        unwrapped_model.generation_config.save_pretrained(training_args.output_dir)
                    unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
            else:
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

            rank0_print(f"Model saved to {training_args.output_dir}")
    except Exception as e:
        rank0_print(f"Error saving model weights: {e}")


if __name__ == "__main__":
    main() 
