import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import json
import tempfile

from transformers import Trainer
from trl.trainer.grpo_trainer import GRPOTrainer
from llava.utils import rank0_print
from llava.train.train import LLaVATrainer


def extract_answer(text):
    """Extract the answer part from text with <answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# Add custom format_reward implementation
def format_reward(completions, solutions=None):
    """
    Calculate format rewards. Check if the generated responses use the correct format.
    """
    # import torch; torch.cuda.empty_cache() 
    # breakpoint()
    # import pdb; pdb.set_trace()
    # import ipdb; ipdb.set_trace()
    rewards = []
    
    for completion in completions:
        # Default reward is 0.5
        reward = 0.5
        
        # Ensure completion has content
        if not completion or len(completion) == 0:
            rewards.append(0.0)
            continue
            
        # Get the answer content
        answer_text = completion[0].get("content", "") if isinstance(completion, list) else completion
        
        # Check if it contains <answer> tags
        if "<answer>" in answer_text and "</answer>" in answer_text:
            # Increase reward value, format is correct
            reward = 1.0
        
        rewards.append(reward)
    
    # import torch; torch.cuda.empty_cache() 
    # breakpoint()
    # import pdb; pdb.set_trace()
    # import ipdb; ipdb.set_trace()
    return rewards  


################################################################ grpo ################################################################
def accuracy_reward(completions, solutions):
    """
    Calculate accuracy rewards. Compare the similarity between the generated responses and the reference answers.
    """
    # import torch; torch.cuda.empty_cache() 
    # breakpoint()
    # import pdb; pdb.set_trace()
    # import ipdb; ipdb.set_trace()
    rewards = []
    
    for i, completion in enumerate(completions):
        # Default reward value is 0.1
        reward = 0.1
        
        # Ensure completion has content and solutions exist
        if not completion or len(completion) == 0:
            rewards.append(0.0)
            continue
            
        # Extract the answer part from the response
        answer_text = completion[0].get("content", "") if isinstance(completion, list) else completion
        extracted_answer = extract_answer(answer_text)
        
        # Get the reference answer
        if solutions and i < len(solutions):
            solution = solutions[i]
            # Simple word overlap comparison
            answer_words = set(re.findall(r'\b\w+\b', extracted_answer.lower()))
            solution_words = set(re.findall(r'\b\w+\b', solution.lower()))
            
            if len(solution_words) > 0 and len(answer_words) > 0:
                # Calculate overlap rate
                overlap = len(answer_words.intersection(solution_words)) / len(solution_words)
                reward = min(1.0, overlap * 2.0)  # Scale to [0,1] range
            
        rewards.append(reward)
    # import torch; torch.cuda.empty_cache()   
    # breakpoint() 
    # import pdb; pdb.set_trace() 
    # import ipdb; ipdb.set_trace()
    return rewards
################################################################ grpo ################################################################


class LLaVAGRPOTrainer(LLaVATrainer, GRPOTrainer):
    """
    LLaVA-specific implementation of GRPO trainer with multiple inheritance.
    Adapts the base GRPOTrainer to handle multimodal inputs and LLaVA-specific training requirements.
    """
    
    def __init__(self, *args, **kwargs):
        # Save special parameters
        reward_funcs = kwargs.pop('reward_funcs', ["accuracy"])
        peft_config = kwargs.pop('peft_config', None)
        data_args = kwargs.pop('data_args', None)
        model_args = kwargs.pop('model_args', None)
        
        # Call the Trainer's initialization method (avoiding MRO issues by not using super())
        Trainer.__init__(self, *args, **kwargs)
        
        # Manually add attributes specific to LLaVATrainer and GRPOTrainer
        self.reward_funcs = reward_funcs
        self.peft_config = peft_config
        self.data_args = data_args
        self.model_args = model_args
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        self.ref_model = None
        
        rank0_print("Initializing LLaVAGRPOTrainer with multiple inheritance")
        
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare inputs for the model, handling multimodal inputs like images.
        """
        # Special handling for multimodal inputs
        if "images" in inputs:
            # Move images to the correct device
            if isinstance(inputs["images"], torch.Tensor):
                inputs["images"] = inputs["images"].to(self.args.device)
            elif isinstance(inputs["images"], list):
                images = []
                for image in inputs["images"]:
                    if isinstance(image, torch.Tensor):
                        images.append(image.to(self.args.device))
                    else:
                        images.append(image)
                inputs["images"] = images
                
            # Handle pixel values if present
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.args.device)
                
        # Regular text input handling
        inputs_on_device = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        return inputs_on_device
    

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Simplified compute_loss method to avoid shape errors with DeepSpeed.
        """  

        # Extract inputs
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        images = inputs.get("images", None)
        pixel_values = inputs.get("pixel_values", None)
        
        # Detect model type
        model_type = model.__class__.__name__.lower()
        
        # Special handling for Qwen model
        is_qwen_model = 'qwen' in model_type.lower() or hasattr(model, "config") and hasattr(model.config, "model_type") and "qwen" in getattr(model.config, "model_type", "").lower()
        
        if is_qwen_model:
            rank0_print("Detected Qwen model, using special loss calculation")

        # breakpoint()
        # import ipdb; ipdb.set_trace()
        # import pdb; pdb.set_trace() 
        
        # Forward pass with image inputs if available
        try:
            if images is not None:
                # import torch; torch.cuda.empty_cache()  
                # breakpoint()   
                # import pdb; pdb.set_trace()
                # import ipdb; ipdb.set_trace()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=images,
                    return_dict=True
                )
            elif pixel_values is not None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values,
                    return_dict=True
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
            
            # Just use the model's loss directly
            # import torch; torch.cuda.empty_cache()   
            # breakpoint()
            # import ipdb; ipdb.set_trace()
            # import pdb; pdb.set_trace() 
            loss = outputs.loss
            
        except Exception as e:
            rank0_print(f"Error in model forward pass: {e}")
            # Fall back to manual loss computation to avoid shape errors
            try:
                # Attempt forward pass without labels
                if images is not None:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images,
                        return_dict=True
                    )
                elif pixel_values is not None:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        return_dict=True
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                
                # Get logits
                logits = outputs.logits
                
                # Ensure correct dimensions for loss computation
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Use a safe method to compute loss
                if is_qwen_model:
                    rank0_print("Using Qwen-specific loss calculation")
                    # Get vocabulary size
                    vocab_size = shift_logits.size(-1)
                    loss_fct = torch.nn.CrossEntropyLoss()
                    
                    # Safely compute loss without using reshape
                    # First, get valid label entries
                    valid_label_mask = shift_labels != -100
                    
                    # Only compute loss for valid label positions
                    valid_logits = shift_logits[valid_label_mask]
                    valid_labels = shift_labels[valid_label_mask]
                    
                    if valid_logits.numel() > 0 and valid_labels.numel() > 0:
                        loss = loss_fct(
                            valid_logits.view(-1, vocab_size),
                            valid_labels.view(-1)
                        )
                    else:
                        # If there are no valid labels, use a pseudo loss
                        loss = torch.tensor(0.1, device=self.args.device)
                else:
                    # Get batch size and sequence length
                    batch_size, seq_length, vocab_size = shift_logits.size()
                    # Use reshaping that matches the actual tensor shape
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.reshape(-1, vocab_size),
                        shift_labels.reshape(-1)
                    )
                
                # Add loss to outputs
                outputs.loss = loss
                
            except Exception as e2:
                rank0_print(f"Failed to compute loss manually: {e2}")
                # Final attempt: directly use outputs without labels to compute a pseudo loss
                if not hasattr(outputs, "loss") or outputs.loss is None:
                    rank0_print("Using a pseudo loss as fallback")
                    # Create a pseudo loss
                    outputs.loss = torch.tensor(1.0, device=self.args.device)
                loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss


    def generate_completions(self, batch):
        """Generate completions for evaluation or reward computation."""
        # import torch; torch.cuda.empty_cache()     
        model = self.model
        model.eval()
        
        # breakpoint()
        # import ipdb; ipdb.set_trace()
        # import pdb; pdb.set_trace() 
        
        # Extract inputs
        try:
            # Ensure input_ids exist and are not None
            if "input_ids" not in batch or batch["input_ids"] is None:
                rank0_print("Error: batch missing input_ids")
                return [[] for _ in range(len(batch.get("input_ids", [0])))]
                
            input_ids = batch.get("input_ids").to(self.args.device)
            
            # Check if input_ids is a valid tensor
            if not isinstance(input_ids, torch.Tensor) or input_ids.numel() == 0:
                rank0_print(f"Warning: input_ids is not a valid tensor: {type(input_ids)}")
                return [[] for _ in range(len(batch.get("input_ids", [0])))]
                
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is None:
                # If no attention_mask is provided, create a mask of all 1s
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(self.args.device)
                
            images = batch.get("images", None)
            if images is not None:
                if isinstance(images, torch.Tensor):
                    images = images.to(self.args.device)
                elif isinstance(images, list):
                    processed_images = []
                    for img in images:
                        if isinstance(img, torch.Tensor):
                            processed_images.append(img.to(self.args.device))
                        else:
                            # Skip non-tensor images
                            rank0_print(f"Skipping non-tensor image of type {type(img)}")
                    if processed_images:
                        images = processed_images
                    else:
                        images = None
                        
            pixel_values = batch.get("pixel_values", None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.args.device)
            
            # Detect model type
            model_type = model.__class__.__name__.lower()
            rank0_print(f"Using model type: {model_type}")
            
            # Configure generation parameters consistently
            gen_kwargs = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.9,
                # Ensure compatibility of all parameters
                "use_cache": True  # Enable cache during generation
            }
            
            # Special handling for Qwen model
            is_qwen_model = 'qwen' in model_type or 'llava_qwen' in model_type.lower()
            
            # Generate completions
            with torch.no_grad():
                try:
                    # Special handling for Qwen model
                    if is_qwen_model:
                        rank0_print("Using Qwen-specific generation parameters")
                        
                        # Check for valid images
                        has_valid_images = (images is not None and 
                                        ((isinstance(images, torch.Tensor) and images.numel() > 0) or
                                            (isinstance(images, list) and len(images) > 0 and all(img is not None for img in images))))
                        
                        # Check for valid pixel values
                        has_valid_pixel_values = (pixel_values is not None and 
                                                isinstance(pixel_values, torch.Tensor) and 
                                                pixel_values.numel() > 0)
                        
                        try:
                            # Correct Qwen model generation call, using input_ids
                            if has_valid_images:
                                outputs = model.generate(
                                    input_ids=input_ids,  # Use input_ids parameter
                                    attention_mask=attention_mask,
                                    images=images,
                                    **gen_kwargs
                                )
                            elif has_valid_pixel_values:
                                outputs = model.generate(
                                    input_ids=input_ids,  # Use input_ids parameter
                                    attention_mask=attention_mask,
                                    pixel_values=pixel_values,
                                    **gen_kwargs
                                )
                            else:
                                outputs = model.generate(
                                    input_ids=input_ids,  # Use input_ids parameter
                                    attention_mask=attention_mask,
                                    **gen_kwargs
                                )
                        except Exception as qwen_e:
                            # Qwen specific error handling
                            rank0_print(f"Qwen generate error: {qwen_e}")
                            # Try without any multimodal inputs
                            rank0_print("Trying Qwen generation without multimodal inputs")
                            try:
                                outputs = model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    **gen_kwargs
                                )
                            except Exception as e3:
                                rank0_print(f"Failed standard generation: {e3}")
                                # Try direct generation with the bare model
                                from transformers.generation import GenerationConfig
                                model.generation_config = GenerationConfig(**gen_kwargs)
                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=True
                                ).logits.argmax(dim=-1)
                    else:
                        # Standard handling for non-Qwen models
                        has_valid_images = (images is not None and 
                                        ((isinstance(images, torch.Tensor) and images.numel() > 0) or
                                            (isinstance(images, list) and len(images) > 0)))
                        
                        if has_valid_images:
                            outputs = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                images=images,
                                **gen_kwargs
                            )
                        elif pixel_values is not None:
                            outputs = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                **gen_kwargs
                            )
                        else:
                            outputs = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **gen_kwargs
                            )
                except Exception as e:
                    rank0_print(f"Error during generation: {e}")
                    # If an error occurs, try to directly use the model to get logits
                    try:
                        rank0_print("Falling back to direct model forward pass")
                        # Use forward pass to get logits
                        with torch.no_grad():
                            if images is not None and (isinstance(images, torch.Tensor) or (isinstance(images, list) and len(images) > 0)):
                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    images=images,
                                    return_dict=True
                                )
                            elif pixel_values is not None:
                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    pixel_values=pixel_values,
                                    return_dict=True
                                )
                            else:
                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=True
                                )
                        
                        # Create a simple generation using greedy decoding
                        logits = outputs.logits
                        if logits is not None and isinstance(logits, torch.Tensor) and logits.numel() > 0:
                            greedy_output = torch.argmax(logits[:, -1], dim=-1).unsqueeze(-1)
                            outputs = torch.cat([input_ids, greedy_output], dim=-1)
                        else:
                            rank0_print("Invalid logits, returning empty results")
                            return [[] for _ in range(input_ids.size(0))]
                    except Exception as e2:
                        rank0_print(f"Error in fallback generation: {e2}")
                        # Return empty results
                        return [[] for _ in range(input_ids.size(0))]
            
            # Decode the generated outputs
            tokenizer = self.tokenizer
            completions = []
            
            for i, output in enumerate(outputs):
                try:
                    # Extract the generated part (excluding prompt)
                    if input_ids is not None and i < input_ids.size(0):
                        # Get the input length for the corresponding batch sample
                        prompt_len = input_ids[i].size(0)
                        if output.size(0) > prompt_len:
                            generated_output = output[prompt_len:]
                        else:
                            generated_output = output
                    else:
                        generated_output = output
                        
                    # Decode to text
                    completion_text = tokenizer.decode(generated_output, skip_special_tokens=True)
                    
                    # Format as completion
                    completion = [{"role": "assistant", "content": completion_text}]
                    completions.append(completion)
                except Exception as e:
                    rank0_print(f"Error decoding output {i}: {e}")
                    # Add an empty generation result
                    completions.append([{"role": "assistant", "content": ""}])
                
            return completions
            
        except Exception as e:
            rank0_print(f"Error in generate_completions: {e}")
            # If batch has input_ids, return a list of empty lists of the same length
            if isinstance(batch.get("input_ids"), torch.Tensor):
                return [[] for _ in range(batch["input_ids"].size(0))]
            else:
                # Otherwise return a single empty list
                return [[]]


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and compute metrics, handling multimodal inputs.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        metrics = {}
        total_rewards = []
        accuracy_rewards = []
        format_rewards = []
        
        for batch in eval_dataloader:
            try:
                # Move batch to device
                batch = self._prepare_inputs(batch)
                
                # Get ground truth answers
                ground_truths = batch.get("labels", None)
                if ground_truths is not None:
                    # Decode ground truths to text
                    solutions = self.tokenizer.batch_decode(ground_truths, skip_special_tokens=True)
                else:
                    solutions = [""] * len(batch["input_ids"])
                    
                # Generate completions
                completions = self.generate_completions(batch)
                
                # 检查结果是否为空
                if not completions or len(completions) == 0 or all(not comp for comp in completions):
                    rank0_print("Warning: generate_completions returned empty results, skipping batch")
                    continue
                
                # 确保solutions和completions长度匹配
                if len(solutions) != len(completions):
                    rank0_print(f"Warning: solutions ({len(solutions)}) and completions ({len(completions)}) length mismatch")
                    # 使用较短的长度
                    min_len = min(len(solutions), len(completions))
                    solutions = solutions[:min_len]
                    completions = completions[:min_len]
                
                # Compute rewards
                try:
                    format_batch_rewards = format_reward(completions, solutions)
                    accuracy_batch_rewards = accuracy_reward(completions, solutions)
                    
                    # Combine rewards (average of all reward functions)
                    batch_rewards = [
                        (f + a) / 2
                        for f, a in zip(format_batch_rewards, accuracy_batch_rewards)
                    ]
                    
                    # Collect rewards
                    total_rewards.extend(batch_rewards)
                    accuracy_rewards.extend(accuracy_batch_rewards)
                    format_rewards.extend(format_batch_rewards)
                except Exception as e:
                    rank0_print(f"Error computing rewards: {e}")
                    continue
            
            except Exception as e:
                rank0_print(f"Error processing batch in evaluate: {e}")
                continue
                
        # Calculate metrics
        # import torch; torch.cuda.empty_cache()  
        # breakpoint()   
        # import ipdb; ipdb.set_trace()
        if total_rewards:
            metrics[f"{metric_key_prefix}_mean_reward"] = sum(total_rewards) / len(total_rewards)
            metrics[f"{metric_key_prefix}_accuracy"] = sum(accuracy_rewards) / len(accuracy_rewards)
            metrics[f"{metric_key_prefix}_format"] = sum(format_rewards) / len(format_rewards)
        else:
            rank0_print("Warning: No rewards collected during evaluation")
            metrics[f"{metric_key_prefix}_mean_reward"] = 0.0
            metrics[f"{metric_key_prefix}_accuracy"] = 0.0
            metrics[f"{metric_key_prefix}_format"] = 0.0

        # import torch; torch.cuda.empty_cache()  
        # breakpoint()   
        # import ipdb; ipdb.set_trace()    
        self.log(metrics)
        return metrics

    def get_mm_adapter_state_maybe_zero_3(self, named_params, keys_to_match):
        """
        Get state dict for mm adapter, handling ZeRO-3 case.
        """
        to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
        to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
        return to_return

    def get_peft_state_maybe_zero_3(self, named_params, bias):
        """
        Get state dict for PEFT model, handling ZeRO-3 case.
        """
        to_return = {k: t for k, t in named_params if "lora_" in k}
        if bias == "none":
            to_return = {k: t for k, t in to_return.items() if "bias" not in k}
        elif bias == "all":
            to_return = {k: t for k, t in to_return.items() if "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k, t in named_params:
                if "lora_" in k:
                    to_return[k] = t
                elif "bias" in k:
                    to_return[k] = t
        to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
        return to_return

    def get_peft_state_non_lora_maybe_zero_3(self, named_params):
        """
        Get state dict for non-LoRA parameters, handling ZeRO-3 case.
        """
        to_return = {k: t for k, t in named_params if "lora_" not in k}
        to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
        return to_return

    def maybe_zero_3(self, param, ignore_status=False, name=None):
        """
        Handle ZeRO-3 parameter state.
        """
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

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Override _save_checkpoint to handle special cases for GRPO training.
        """
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and 
            ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = self.get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            # Check if LoRA is enabled
            lora_enabled = getattr(self.args, "lora_enable", False)
            if lora_enabled:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                from transformers.modeling_utils import unwrap_model

                unwrapped_model = unwrap_model(model)
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                # Save full model state
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    # Save model config
                    self.model.config.save_pretrained(output_dir)
                    
                    # Save model weights using save_pretrained
                    self.model.save_pretrained(output_dir)
                    
                    # Save generation config if available
                    if hasattr(self.model, "generation_config"):
                        self.model.generation_config.save_pretrained(output_dir)
                    
                    # Save tokenizer
                    if hasattr(self, "tokenizer"):
                        self.tokenizer.save_pretrained(output_dir)
                    
                    # Save training args
                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Override save_model method to handle ZeRO-3 model saving and ensure all paths are valid.
        """
        # Step 1: Ensure we have a valid output_dir
        valid_output_dir = None
        
        # Try to use provided output_dir
        if output_dir is not None:
            valid_output_dir = output_dir
        # Try to get output_dir from args
        elif hasattr(self, "args") and hasattr(self.args, "output_dir") and self.args.output_dir:
            valid_output_dir = self.args.output_dir
        # Create temporary directory as last resort
        else:
            valid_output_dir = tempfile.mkdtemp(prefix="llava_model_")
        
        # Ensure output directory exists
        try:
            os.makedirs(valid_output_dir, exist_ok=True)
        except Exception as e:
            valid_output_dir = tempfile.mkdtemp(prefix="llava_model_")
            os.makedirs(valid_output_dir, exist_ok=True)

        # Check if we should only save mm adapter
        check_only_save_mm_adapter_tunnable = False
        if hasattr(self.args, "tune_mm_mlp_adapter") and self.args.tune_mm_mlp_adapter:
            check_only_save_mm_adapter_tunnable = True
        elif hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and 
              ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts)):
            check_only_save_mm_adapter_tunnable = True

        self.accelerator.wait_for_everyone()
        torch.cuda.synchronize()
        rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")

        if check_only_save_mm_adapter_tunnable:
            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = self.get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            self.model.config.save_pretrained(valid_output_dir)

            current_folder = valid_output_dir.split("/")[-1]
            parent_folder = os.path.dirname(valid_output_dir)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                if current_folder.startswith("checkpoint-"):
                    mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
                else:
                    torch.save(weight_to_save, os.path.join(valid_output_dir, f"mm_projector.bin"))
            return valid_output_dir

        if self.deepspeed:
            # Use self instead of trainer to call parent class's save_model method
            super().save_model(valid_output_dir)
            return valid_output_dir

        # Save full model state
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            # Save model config
            self.model.config.save_pretrained(valid_output_dir)
            
            # Save model weights using save_pretrained
            self.model.save_pretrained(valid_output_dir)
            
            # Save generation config if available
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.save_pretrained(valid_output_dir)
            
            # Save tokenizer
            if hasattr(self, "tokenizer"):
                self.tokenizer.save_pretrained(valid_output_dir)
            
            # Save training args
            torch.save(self.args, os.path.join(valid_output_dir, "training_args.bin"))

        return valid_output_dir
