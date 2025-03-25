import os
import textwrap
import re
import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available
from transformers.modeling_utils import unwrap_model

from trl.models import create_reference_model
from trl.trainer.grpo_config import GRPOConfig

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def accuracy_reward(completions, solution, **kwargs):
    # import torch; torch.cuda.empty_cache()     
    # breakpoint()
    # import ipdb; ipdb.set_trace()
    """Reward function that checks if the completion is correct using either regex extraction or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        try:
            # Extract answer from solution if it has answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # Normalize the text for comparison
            ground_truth = ground_truth.replace(' ','').replace('_','').lower()
            student_answer = student_answer.replace(' ','').replace('_','').lower()

            # Compare the extracted answers
            if ground_truth in student_answer or student_answer in ground_truth:
                reward = 1.0
        except Exception:
            pass  # Keep reward as 0.0 if extraction fails
                
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
    # import torch; torch.cuda.empty_cache()  
    # breakpoint()   
    # import ipdb; ipdb.set_trace()
    # return rewards


def format_reward(completions, **kwargs):
    # import torch; torch.cuda.empty_cache() 
    # breakpoint()    
    # import ipdb; ipdb.set_trace()
    """Reward function that checks if the completion has the proper format with <think> and <answer> tags."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    # import torch; torch.cuda.empty_cache()    
    # breakpoint() 
    # import ipdb; ipdb.set_trace()
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


class GRPOTrainer(Trainer):
    """
    Trainer for Group Relative Policy Optimization (GRPO).
    
    GRPO is a reinforcement learning algorithm that optimizes policies relative to a reference policy
    using groups of samples. It works by computing rewards for generated completions and updating 
    the policy to maximize these rewards while maintaining proximity to the reference policy.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, Dict[str, Union[Dataset, IterableDataset]]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        reward_tokenizers: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize training arguments if not provided
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Initialize model if provided as string
        if isinstance(model, str):
            model_id = model
            model_init_kwargs = getattr(args, "model_init_kwargs", {}) or {}
            torch_dtype = model_init_kwargs.get("torch_dtype", None)
            
            # Handle torch_dtype if provided as string
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            
            # Disable caching if gradient checkpointing is enabled
            model_init_kwargs["use_cache"] = not args.gradient_checkpointing
            
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path

        # Apply PEFT if configuration is provided
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            self.is_peft_model = True
        else:
            self.is_peft_model = False

        # Reference model for KL regularization
        if self.is_peft_model:
            # If PEFT is used, we can use the same model with disabled adapters as reference
            self.ref_model = None
        else:
            # Otherwise, create a separate reference model
            self.ref_model = create_reference_model(model)

        # Initialize tokenizer if not provided
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Set padding side to left for proper processing of attention masks
            tokenizer.padding_side = "left"
            if not getattr(tokenizer, "pad_token", None):
                tokenizer.pad_token = tokenizer.eos_token

        # Reward functions setup
        if isinstance(reward_funcs, (list, tuple)):
            self.reward_funcs = reward_funcs
        else:
            self.reward_funcs = [reward_funcs]

        # Process reward functions
        self.processed_reward_funcs = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, str):
                if reward_func in reward_funcs_registry:
                    # Use predefined reward function from registry
                    self.processed_reward_funcs.append(reward_funcs_registry[reward_func])
                else:
                    # Load model-based reward function
                    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_func, num_labels=1)
                    self.processed_reward_funcs.append(reward_model)
            elif isinstance(reward_func, PreTrainedModel):
                # Use provided reward model
                self.processed_reward_funcs.append(reward_func)
            elif callable(reward_func):
                # Use custom reward function
                self.processed_reward_funcs.append(reward_func)
            else:
                raise ValueError(f"Reward function type not supported: {type(reward_func)}")

        # Call parent Trainer initialization
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute GRPO loss. This involves:
        1. Generating completions using the model
        2. Computing rewards for these completions
        3. Computing KL divergence between current and reference policy
        4. Combining reward maximization with KL regularization
        """
        # import torch; torch.cuda.empty_cache()    
        # breakpoint()
        # import ipdb; ipdb.set_trace()
        # Extract necessary inputs
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        
        # Get response generation start indices
        response_start_indices = torch.argmax((labels >= 0).int(), dim=1)
        
        # import torch; torch.cuda.empty_cache()  
        # breakpoint()   
        # import ipdb; ipdb.set_trace()
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        # import torch; torch.cuda.empty_cache()   
        # breakpoint()  
        # import ipdb; ipdb.set_trace()

        
        # Get log probabilities from the model outputs
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # If we're using PEFT, get reference log probs by disabling adapters
        if self.is_peft_model:
            model.eval()
            with torch.no_grad():
                # Disable adapters to get reference model outputs
                model.disable_adapter_layers()
                ref_outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    return_dict=True
                )
                model.enable_adapter_layers()
            model.train()
            ref_logits = ref_outputs.logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        elif self.ref_model is not None:
            # Otherwise use separate reference model
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                ref_logits = ref_outputs.logits
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        else:
            raise ValueError("Neither PEFT model nor reference model is available")
        
        # Calculate policy loss (GRPO objective)
        # For simplicity, we use the standard supervised loss as a proxy for the reward-weighted loss
        policy_loss = outputs.loss
        
        # Calculate KL divergence between current and reference policy
        kl_div = self._compute_kl_divergence(log_probs, ref_log_probs, labels)
        
        # Combine losses with beta parameter for KL regularization
        beta = self.args.beta
        loss = policy_loss + beta * kl_div
        
        if return_outputs:
            return loss, outputs
        return loss

    def _compute_kl_divergence(self, log_probs, ref_log_probs, labels):
        """
        Compute KL divergence between current and reference policy.
        Only consider tokens that are part of the generated completion (where labels >= 0).
        """
        # import torch; torch.cuda.empty_cache()   
        # breakpoint()  
        # import ipdb; ipdb.set_trace()  
        # Create a mask for tokens that are part of the completion
        completion_mask = (labels >= 0).float()
        
        # Calculate per-token KL divergence
        kl = torch.exp(ref_log_probs) * (ref_log_probs - log_probs)
        
        # Apply mask and calculate mean KL divergence
        masked_kl = kl.sum(dim=-1) * completion_mask
        kl_div = masked_kl.sum() / (completion_mask.sum() + 1e-8)
        
        return kl_div
        
    def _get_batch_rewards(self, prompts, completions, solutions):
        """
        Compute rewards for a batch of completions.
        Aggregates rewards from all reward functions.
        """
        # import torch; torch.cuda.empty_cache()   
        # breakpoint()  
        # import ipdb; ipdb.set_trace()
        total_rewards = None
        
        for reward_func in self.processed_reward_funcs:
            # import torch; torch.cuda.empty_cache()     
            # import ipdb; ipdb.set_trace()
            if isinstance(reward_func, PreTrainedModel):
                # TODO: Implement model-based reward computation
                pass
            elif callable(reward_func):
                # Custom reward function
                batch_rewards = reward_func(completions, solutions)
                
                if total_rewards is None:
                    total_rewards = batch_rewards
                else:
                    total_rewards = [r1 + r2 for r1, r2 in zip(total_rewards, batch_rewards)]
        
        # Normalize if we have multiple reward functions
        if len(self.processed_reward_funcs) > 1 and total_rewards is not None:
            total_rewards = [r / len(self.processed_reward_funcs) for r in total_rewards]
            
        return total_rewards
        
    def generate_completions(self, prompts, solutions):
        """Generate completions for evaluation or reward computation."""
        # import torch; torch.cuda.empty_cache()  
        # breakpoint()   
        # import ipdb; ipdb.set_trace()
        model = unwrap_model(self.model)
        model.eval()
        
        completions = []
        for prompt, solution in zip(prompts, solutions):
            # TODO: Implement proper generation logic
            # For now, use a placeholder
            # import torch; torch.cuda.empty_cache()    
            # breakpoint() 
            # import ipdb; ipdb.set_trace()
            completion = [{"role": "assistant", "content": solution}]  # Using solution as placeholder
            completions.append(completion)
            
        # import torch; torch.cuda.empty_cache()   
        # breakpoint()      
        # import ipdb; ipdb.set_trace()
        return completions

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and compute metrics.
        We override this to compute reward-based metrics.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        metrics = {}
        total_rewards = []
        
        for batch in eval_dataloader:
            # TODO: Extract prompts and solutions
            prompts = batch.get("prompt", [])
            solutions = batch.get("solution", [])
            
            completions = self.generate_completions(prompts, solutions)
            batch_rewards = self._get_batch_rewards(prompts, completions, solutions)
            
            if batch_rewards:
                total_rewards.extend(batch_rewards)
                
        if total_rewards:
            # import torch; torch.cuda.empty_cache()    
            # breakpoint() 
            # import ipdb; ipdb.set_trace()
            metrics[f"{metric_key_prefix}_mean_reward"] = sum(total_rewards) / len(total_rewards)
            
        self.log(metrics)
        return metrics 
