# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available. LoRA fine-tuning will be disabled.")

from opentslm.model_config import ENCODER_OUTPUT_DIM
from .TimeSeriesLLM import TimeSeriesLLM
from ..encoder.TransformerCNNEncoder import TransformerCNNEncoder
from ..projector.MLPProjector import MLPProjector
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


class OpenTSLMClassifier(TimeSeriesLLM):
    """
    OpenTSLM with classification head instead of generative head.
    
    Experiment A: Isolate LLM's role as sequence aggregator.
    
    Architecture:
        [PrePrompt] + [TS Tokens] + [PostPrompt] + [ANS]
                                                     ↓
                                          classification_head
                                                     ↓
                                                num_classes
    
    Key differences from OpenTSLMSP:
    - Adds learnable [ANS] query token at the end
    - Uses classification head (CrossEntropyLoss) instead of language modeling head
    - No token generation or label tokenization
    """
    
    def __init__(
        self,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        num_classes: int = 2,
        device: str = "cuda",
        encoder_type: str = "transformer_cnn",
        encoder_pretrained_path: Optional[str] = None,
        tslanet_config: Optional[Dict] = None,
    ):
        super().__init__(device)
        
        self.num_classes = num_classes

        # 1) tokenizer (ensure pad_token exists)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="eager",
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 3) encoder + projector (same as OpenTSLMSP)
        self.encoder_type = encoder_type
        if encoder_type == "tslanet":
            from ..encoder.TSLANetEncoder import TSLANetEncoder
            config = tslanet_config or {}
            default_config = {
                "output_dim": ENCODER_OUTPUT_DIM,
                "patch_size": 8,
                "emb_dim": 128,
                "depth": 2,
                "dropout": 0.15,
            }
            default_config.update(config)
            self.encoder = TSLANetEncoder(**default_config).to(device)
            self.patch_size = default_config.get("patch_size", 8)
            
            if encoder_pretrained_path:
                self.encoder.load_pretrained(encoder_pretrained_path)
                print(f"✅ Loaded TSLANet pretrained weights from: {encoder_pretrained_path}")
        else:
            self.encoder = TransformerCNNEncoder().to(device)
            self.patch_size = 4
        
        self.projector = MLPProjector(
            ENCODER_OUTPUT_DIM, self.llm.config.hidden_size, device=device
        ).to(device)

        # 4) Learnable [ANS] query token
        hidden_size = self.llm.config.hidden_size
        self.ans_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        
        # 5) Classification head
        self.classification_head = nn.Linear(hidden_size, num_classes).to(device)
        
        # LoRA-related attributes
        self.lora_enabled = False
        self.original_llm = None

        # Freeze the LLM backbone initially
        for p in self.llm.parameters():
            p.requires_grad = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for the LLM to reduce memory usage."""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled for LLM")
        else:
            print("⚠️ LLM does not support gradient_checkpointing_enable()")

    def forward(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        Forward pass for DDP compatibility.
        
        Args:
            batch: List of dictionaries containing the batch data.
                   Each dict should have 'label' key with integer class label.
            
        Returns:
            Loss tensor (CrossEntropyLoss)
        """
        labels = torch.tensor([item["label"] for item in batch], device=self.device, dtype=torch.long)
        return self.compute_loss(batch, labels)

    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        """Enable LoRA fine-tuning for the LLM component."""
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "peft package is required for LoRA fine-tuning. Please install with: pip install peft"
            )

        if self.lora_enabled:
            raise RuntimeError(
                "LoRA is already enabled. Call disable_lora() first if you want to reconfigure LoRA."
            )

        self.original_llm = self.llm

        if target_modules is None:
            target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        try:
            self.llm = get_peft_model(self.llm, lora_config)
            self.lora_enabled = True

            lora_params = sum(
                p.numel()
                for name, p in self.llm.named_parameters()
                if p.requires_grad and "lora_" in name
            )
            trainable_params = sum(
                p.numel() for p in self.llm.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.llm.parameters())
            print(f"✅ LoRA enabled:")
            print(f"   LoRA parameters: {lora_params:,}")
            print(f"   Total trainable parameters: {trainable_params:,}")
            print(f"   Total parameters: {total_params:,}")
            print(f"   LoRA %: {100 * lora_params / total_params:.2f}%")
            print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")

        except Exception as e:
            print(f"❌ Failed to enable LoRA: {e}")
            raise

    def get_lora_parameters(self):
        """Get LoRA parameters for the optimizer."""
        if not self.lora_enabled:
            return []

        lora_params = []
        for name, param in self.llm.named_parameters():
            if param.requires_grad and "lora_" in name:
                lora_params.append(param)
        return lora_params

    def disable_lora(self):
        """Disable LoRA and revert to original frozen LLM."""
        if not self.lora_enabled:
            raise RuntimeError(
                "LoRA is not enabled. Cannot disable LoRA when it's not active."
            )

        if self.original_llm is not None:
            self.llm = self.original_llm
            self.original_llm = None

        self.lora_enabled = False
        print("✅ LoRA disabled, reverted to frozen LLM")

    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch and add [ANS] token at the end.
        
        This is identical to OpenTSLMSP's pad_and_apply_batch, but adds
        the learnable [ANS] query token at the end of each sequence.
        
        Returns (inputs_embeds, attention_mask) with [ANS] appended.
        """
        device = self.device
        H = self.llm.config.hidden_size

        # 1) Gather all texts
        all_texts: List[str] = []
        text_ptrs: List[Tuple[int, int]] = []
        ts_counts: List[int] = []
        for sample in batch:
            start = len(all_texts)
            all_texts.append(sample["pre_prompt"])
            all_texts.extend(sample["time_series_text"])
            all_texts.append(sample["post_prompt"])
            end = len(all_texts)
            text_ptrs.append((start, end))
            ts_counts.append(len(sample["time_series_text"]))

        # 2) Tokenize & embed all texts
        tok = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tok.input_ids.to(device, non_blocking=True)
        attn_mask = tok.attention_mask.to(device, non_blocking=True)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 3) Batch time-series encode & project
        ts_list: List[torch.Tensor] = []
        for sample in batch:
            for ts in sample["time_series"]:
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                ts_list.append(ts)

        if ts_list:
            ts_padded = pad_sequence(ts_list, batch_first=True).to(
                device, non_blocking=True
            )
            T_max = ts_padded.size(1)
            rem = T_max % self.patch_size
            if rem:
                pad_len = self.patch_size - rem
                pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
                ts_padded = torch.cat([ts_padded, pad], dim=1)

            ts_enc = self.encoder(ts_padded.squeeze(-1))
            ts_proj = self.projector(ts_enc).to(text_embeds.dtype)
        else:
            ts_proj = torch.empty(0, 0, H, device=device, dtype=text_embeds.dtype)

        # 4) Re­assemble per sample
        all_seq_embeds, all_seq_masks = [], []
        ts_offset = 0
        for (start, end), n_ts in zip(text_ptrs, ts_counts):
            sample_embeds = text_embeds[start:end]
            sample_masks = attn_mask[start:end]
            seq_embeds, seq_masks = [], []

            # pre_prompt
            length = sample_masks[0].sum().item()
            seq_embeds.append(sample_embeds[0, :length, :])
            seq_masks.append(sample_masks[0, :length])

            # each (textᵢ, tsᵢ)
            for i in range(n_ts):
                idx = 1 + i
                length = sample_masks[idx].sum().item()
                seq_embeds.append(sample_embeds[idx, :length, :])
                seq_masks.append(sample_masks[idx, :length])

                proj = ts_proj[ts_offset + i]
                seq_embeds.append(proj)
                seq_masks.append(
                    torch.ones(proj.size(0), device=device, dtype=torch.long)
                )

            ts_offset += n_ts

            # post_prompt
            length = sample_masks[-1].sum().item()
            seq_embeds.append(sample_embeds[-1, :length, :])
            seq_masks.append(sample_masks[-1, :length])
            
            # *** ADD [ANS] TOKEN HERE ***
            # Expand ans_token to match dtype
            ans_emb = self.ans_token.to(text_embeds.dtype)  # [1, H]
            seq_embeds.append(ans_emb)
            seq_masks.append(torch.ones(1, device=device, dtype=torch.long))

            all_seq_embeds.append(torch.cat(seq_embeds, dim=0))
            all_seq_masks.append(torch.cat(seq_masks, dim=0))

        # 5) Batch-pad the final sequences
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)  # [B, L_max, H]
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)  # [B, L_max]

        return inputs_embeds, attention_mask

    @torch.no_grad()
    def predict(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        Inference: predict class labels for a batch.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            predicted_classes: Tensor of shape [B] with predicted class indices
        """
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract [ANS] position (last valid token per sample)
        last_hidden = outputs.hidden_states[-1]  # [B, L, H]
        
        # Get the position of last valid token for each sample
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B], -1 for 0-indexing
        batch_size = last_hidden.size(0)
        
        # Extract [ANS] embeddings
        ans_embeds = last_hidden[torch.arange(batch_size), seq_lengths]  # [B, H]
        
        # Classification head
        logits = self.classification_head(ans_embeds)  # [B, num_classes]
        predicted_classes = torch.argmax(logits, dim=-1)  # [B]
        
        return predicted_classes

    def compute_loss(self, batch: List[Dict[str, any]], labels: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            batch: List of sample dictionaries
            labels: Ground truth class labels [B]
            
        Returns:
            CrossEntropyLoss
        """
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract [ANS] position
        last_hidden = outputs.hidden_states[-1]  # [B, L, H]
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_size = last_hidden.size(0)
        
        ans_embeds = last_hidden[torch.arange(batch_size), seq_lengths]  # [B, H]
        
        # Classification head
        logits = self.classification_head(ans_embeds)  # [B, num_classes]
        
        # CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        return loss

    def get_eos_token(self) -> str:
        """For compatibility with data loaders."""
        return self.tokenizer.eos_token

    def save_lora_state_to_checkpoint(self, checkpoint: dict):
        """Save LoRA adapters to checkpoint."""
        checkpoint["lora_enabled"] = self.lora_enabled

        if self.lora_enabled and hasattr(self.llm, "peft_config"):
            try:
                lora_state = {}
                for name, param in self.llm.named_parameters():
                    if param.requires_grad and "lora_" in name:
                        lora_state[name] = param.data.clone()

                if lora_state:
                    checkpoint["lora_state"] = lora_state
                    checkpoint["lora_config"] = self.llm.peft_config
                    print(f"💾 Saved LoRA adapters with {len(lora_state)} parameters")
                    return len(lora_state)
            except Exception as e:
                raise RuntimeError(f"Failed to save LoRA adapters: {e}")

        return 0

    def load_lora_state_from_checkpoint(
        self, checkpoint: dict, allow_missing: bool = False
    ):
        """Load LoRA adapters from checkpoint."""
        checkpoint_has_lora = checkpoint.get("lora_enabled", False)

        if checkpoint_has_lora and "lora_state" in checkpoint:
            if not self.lora_enabled:
                raise RuntimeError(
                    "Checkpoint contains LoRA adapters but LoRA is not currently enabled. "
                    "Call enable_lora() before loading this checkpoint."
                )

            try:
                lora_state = checkpoint["lora_state"]
                loaded_count = 0

                for name, param in self.llm.named_parameters():
                    if name in lora_state and param.requires_grad and "lora_" in name:
                        param.data.copy_(lora_state[name])
                        loaded_count += 1

                print(f"📥 Loaded LoRA adapters: {loaded_count} parameters")
                return loaded_count

            except Exception as e:
                raise RuntimeError(f"Failed to load LoRA adapters: {e}")

        if not checkpoint_has_lora and self.lora_enabled:
            if not allow_missing:
                raise RuntimeError(
                    "Loading checkpoint from before LoRA was enabled, but LoRA is currently enabled. "
                    "LoRA adapters will be randomly initialized. Set allow_missing=True to allow this."
                )
            else:
                print("⚠️  Loading checkpoint from before LoRA was enabled.")
                print("   LoRA adapters will be randomly initialized.")

        return 0
