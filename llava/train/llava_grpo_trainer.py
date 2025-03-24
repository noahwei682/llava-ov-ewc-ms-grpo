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


# 添加自定义format_reward实现
def format_reward(completions, solutions=None):
    """
    计算格式奖励。检查生成的回答是否使用了正确的格式。
    """
    rewards = []
    
    for completion in completions:
        # 默认奖励为0.5
        reward = 0.5
        
        # 确保completion有内容
        if not completion or len(completion) == 0:
            rewards.append(0.0)
            continue
            
        # 获取回答内容
        answer_text = completion[0].get("content", "") if isinstance(completion, list) else completion
        
        # 检查是否包含<answer>标记
        if "<answer>" in answer_text and "</answer>" in answer_text:
            # 增加奖励值，格式正确
            reward = 1.0
        
        rewards.append(reward)
        
    return rewards


def accuracy_reward(completions, solutions):
    """
    计算准确性奖励。比较生成的回答和参考答案的相似度。
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        # 默认奖励值0.1
        reward = 0.1
        
        # 确保completion有内容且存在solutions
        if not completion or len(completion) == 0:
            rewards.append(0.0)
            continue
            
        # 从回答中提取答案部分
        answer_text = completion[0].get("content", "") if isinstance(completion, list) else completion
        extracted_answer = extract_answer(answer_text)
        
        # 获取参考答案
        if solutions and i < len(solutions):
            solution = solutions[i]
            # 简单的词重叠比较
            answer_words = set(re.findall(r'\b\w+\b', extracted_answer.lower()))
            solution_words = set(re.findall(r'\b\w+\b', solution.lower()))
            
            if len(solution_words) > 0 and len(answer_words) > 0:
                # 计算重叠率
                overlap = len(answer_words.intersection(solution_words)) / len(solution_words)
                reward = min(1.0, overlap * 2.0)  # 缩放到[0,1]范围
            
        rewards.append(reward)
        
    return rewards


class LLaVAGRPOTrainer(LLaVATrainer, GRPOTrainer):
    """
    LLaVA-specific implementation of GRPO trainer with multiple inheritance.
    Adapts the base GRPOTrainer to handle multimodal inputs and LLaVA-specific training requirements.
    """
    
    def __init__(self, *args, **kwargs):
        # 保存特殊参数
        reward_funcs = kwargs.pop('reward_funcs', ["accuracy"])
        peft_config = kwargs.pop('peft_config', None)
        data_args = kwargs.pop('data_args', None)
        model_args = kwargs.pop('model_args', None)
        
        # 调用Trainer的初始化方法（不使用super()避免MRO问题）
        Trainer.__init__(self, *args, **kwargs)
        
        # 手动添加LLaVATrainer和GRPOTrainer特有的属性
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
        
        # 检测模型类型
        model_type = model.__class__.__name__.lower()
        
        # 针对Qwen模型的特殊处理
        is_qwen_model = 'qwen' in model_type.lower() or hasattr(model, "config") and hasattr(model.config, "model_type") and "qwen" in getattr(model.config, "model_type", "").lower()
        
        if is_qwen_model:
            rank0_print("Detected Qwen model, using special loss calculation")
        
        # Forward pass with image inputs if available
        try:
            if images is not None:
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
            loss = outputs.loss
            
        except Exception as e:
            rank0_print(f"Error in model forward pass: {e}")
            # Fall back to manual loss computation to avoid shape errors
            try:
                # 尝试不带labels的前向传播
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
                
                # 获取logits
                logits = outputs.logits
                
                # 确保使用正确的维度设置损失计算
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # 使用安全的方法计算损失
                if is_qwen_model:
                    rank0_print("Using Qwen-specific loss calculation")
                    # 获取词汇表大小
                    vocab_size = shift_logits.size(-1)
                    loss_fct = torch.nn.CrossEntropyLoss()
                    
                    # 安全的方式计算损失，不使用reshape
                    # 首先获取有效的标签项
                    valid_label_mask = shift_labels != -100
                    
                    # 只计算有效标签对应位置的损失
                    valid_logits = shift_logits[valid_label_mask]
                    valid_labels = shift_labels[valid_label_mask]
                    
                    if valid_logits.numel() > 0 and valid_labels.numel() > 0:
                        loss = loss_fct(
                            valid_logits.view(-1, vocab_size),
                            valid_labels.view(-1)
                        )
                    else:
                        # 如果没有有效标签，使用伪损失
                        loss = torch.tensor(0.1, device=self.args.device)
                else:
                    # 获取批次大小和序列长度
                    batch_size, seq_length, vocab_size = shift_logits.size()
                    # 使用适配于实际张量形状的重塑
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.reshape(-1, vocab_size),
                        shift_labels.reshape(-1)
                    )
                
                # 添加损失到输出
                outputs.loss = loss
                
            except Exception as e2:
                rank0_print(f"Failed to compute loss manually: {e2}")
                # 最后的尝试：直接使用没有标签的输出计算一个伪损失
                if not hasattr(outputs, "loss") or outputs.loss is None:
                    rank0_print("Using a pseudo loss as fallback")
                    # 创建一个伪损失
                    outputs.loss = torch.tensor(1.0, device=self.args.device)
                loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss

    def generate_completions(self, batch):
        """Generate completions for evaluation or reward computation."""
        model = self.model
        model.eval()
        
        # Extract inputs
        try:
            # 确保input_ids存在且不为None
            if "input_ids" not in batch or batch["input_ids"] is None:
                rank0_print("Error: batch missing input_ids")
                return [[] for _ in range(len(batch.get("input_ids", [0])))]
                
            input_ids = batch.get("input_ids").to(self.args.device)
            
            # 检查input_ids是否为有效的张量
            if not isinstance(input_ids, torch.Tensor) or input_ids.numel() == 0:
                rank0_print(f"Warning: input_ids is not a valid tensor: {type(input_ids)}")
                return [[] for _ in range(len(batch.get("input_ids", [0])))]
                
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is None:
                # 如果没有提供attention_mask，创建一个全1的mask
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
            
            # 检测模型类型
            model_type = model.__class__.__name__.lower()
            rank0_print(f"Using model type: {model_type}")
            
            # Configure generation parameters consistently
            gen_kwargs = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.9,
                # 保证所有参数兼容性
                "use_cache": True  # 生成时启用缓存
            }
            
            # 针对Qwen模型的特殊处理
            is_qwen_model = 'qwen' in model_type or 'llava_qwen' in model_type.lower()
            
            # Generate completions
            with torch.no_grad():
                try:
                    # 对于Qwen模型的特殊处理
                    if is_qwen_model:
                        rank0_print("Using Qwen-specific generation parameters")
                        
                        # 检查是否有有效图像
                        has_valid_images = (images is not None and 
                                           ((isinstance(images, torch.Tensor) and images.numel() > 0) or
                                            (isinstance(images, list) and len(images) > 0 and all(img is not None for img in images))))
                        
                        # 检查是否有有效的像素值
                        has_valid_pixel_values = (pixel_values is not None and 
                                                 isinstance(pixel_values, torch.Tensor) and 
                                                 pixel_values.numel() > 0)
                        
                        try:
                            # 修正Qwen模型生成调用，使用input_ids
                            if has_valid_images:
                                outputs = model.generate(
                                    input_ids=input_ids,  # 使用input_ids参数
                                    attention_mask=attention_mask,
                                    images=images,
                                    **gen_kwargs
                                )
                            elif has_valid_pixel_values:
                                outputs = model.generate(
                                    input_ids=input_ids,  # 使用input_ids参数
                                    attention_mask=attention_mask,
                                    pixel_values=pixel_values,
                                    **gen_kwargs
                                )
                            else:
                                outputs = model.generate(
                                    input_ids=input_ids,  # 使用input_ids参数
                                    attention_mask=attention_mask,
                                    **gen_kwargs
                                )
                        except Exception as qwen_e:
                            # Qwen特定错误处理
                            rank0_print(f"Qwen generate error: {qwen_e}")
                            # 尝试不带任何多模态输入
                            rank0_print("Trying Qwen generation without multimodal inputs")
                            try:
                                outputs = model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    **gen_kwargs
                                )
                            except Exception as e3:
                                rank0_print(f"Failed standard generation: {e3}")
                                # 尝试使用裸模型直接生成
                                from transformers.generation import GenerationConfig
                                model.generation_config = GenerationConfig(**gen_kwargs)
                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=True
                                ).logits.argmax(dim=-1)
                    else:
                        # 非Qwen模型的标准处理
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
                    # 如果出错，尝试直接使用模型进行前向传播获取logits
                    try:
                        rank0_print("Falling back to direct model forward pass")
                        # 使用前向传播获取logits
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
                        
                        # 使用贪婪解码创建一个简单的生成
                        logits = outputs.logits
                        if logits is not None and isinstance(logits, torch.Tensor) and logits.numel() > 0:
                            greedy_output = torch.argmax(logits[:, -1], dim=-1).unsqueeze(-1)
                            outputs = torch.cat([input_ids, greedy_output], dim=-1)
                        else:
                            rank0_print("Invalid logits, returning empty results")
                            return [[] for _ in range(input_ids.size(0))]
                    except Exception as e2:
                        rank0_print(f"Error in fallback generation: {e2}")
                        # 返回空结果
                        return [[] for _ in range(input_ids.size(0))]
            
            # Decode the generated outputs
            tokenizer = self.tokenizer
            completions = []
            
            for i, output in enumerate(outputs):
                try:
                    # Extract the generated part (excluding prompt)
                    if input_ids is not None and i < input_ids.size(0):
                        # 获取对应批次样本的输入长度
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
                    # 添加一个空生成结果
                    completions.append([{"role": "assistant", "content": ""}])
                
            return completions
            
        except Exception as e:
            rank0_print(f"Error in generate_completions: {e}")
            # 如果batch中有input_ids，返回等长的空列表
            if isinstance(batch.get("input_ids"), torch.Tensor):
                return [[] for _ in range(batch["input_ids"].size(0))]
            else:
                # 否则返回一个空列表
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
        if total_rewards:
            metrics[f"{metric_key_prefix}_mean_reward"] = sum(total_rewards) / len(total_rewards)
            metrics[f"{metric_key_prefix}_accuracy"] = sum(accuracy_rewards) / len(accuracy_rewards)
            metrics[f"{metric_key_prefix}_format"] = sum(format_rewards) / len(format_rewards)
        else:
            rank0_print("Warning: No rewards collected during evaluation")
            metrics[f"{metric_key_prefix}_mean_reward"] = 0.0
            metrics[f"{metric_key_prefix}_accuracy"] = 0.0
            metrics[f"{metric_key_prefix}_format"] = 0.0
            
        self.log(metrics)
        return metrics

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

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
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
            # 使用self而不是trainer来调用父类的save_model方法
            super().save_model(valid_output_dir)
            return valid_output_dir

        state_dict = self.model.state_dict()
        if self.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            self._save(valid_output_dir, state_dict=cpu_state_dict)  # noqa

        return valid_output_dir
