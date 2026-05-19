# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available. LoRA fine-tuning will be disabled.")

from opentslm.model_config import ENCODER_OUTPUT_DIM
from .TimeSeriesLLM import TimeSeriesLLM
from ..encoder.NewTSDualBranchEncoder import NewTSDualBranchEncoder
from ..encoder.TransformerCNNEncoder import TransformerCNNEncoder
from ..projector.MLPProjector import MLPProjector
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from .hf_local import resolve_local_hf_snapshot


class OpenTSLMSP(TimeSeriesLLM):
    def __init__(
        self,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
        encoder_type: str = "transformer_cnn",
        encoder_pretrained_path: Optional[str] = None,
        tslanet_config: Optional[Dict[str, Any]] = None,
        newts_dual_branch_config: Optional[Dict[str, Any]] = None,
        llm_attn_impl: str = "sdpa",
    ):
        super().__init__(device)
        self.llm_id = llm_id
        self.llm_source = resolve_local_hf_snapshot(llm_id)
        self.llm_source_is_local = self.llm_source != llm_id
        self.requested_llm_attn_impl = llm_attn_impl

        # 1) tokenizer (ensure pad_token exists)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_source,
            use_fast=True,
            local_files_only=self.llm_source_is_local,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) load LLM
        self.llm, self.llm_attn_impl = self._load_llm_with_attn_fallback(
            llm_id=self.llm_source,
            device=device,
            llm_attn_impl=llm_attn_impl,
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 3) encoder + projector
        self.encoder_type = encoder_type
        (
            self.encoder,
            self.patch_size,
            self.encoder_config,
        ) = self._build_encoder(
            encoder_type=encoder_type,
            device=device,
            encoder_pretrained_path=encoder_pretrained_path,
            tslanet_config=tslanet_config,
            newts_dual_branch_config=newts_dual_branch_config,
        )

        self.projector = MLPProjector(
            ENCODER_OUTPUT_DIM, self.llm.config.hidden_size, device=device
        ).to(device)

        # LoRA-related attributes
        self.lora_enabled = False
        self.original_llm = (
            None  # Keep reference to original model for backward compatibility
        )

        # Freeze the LLM backbone for SP model (internally)
        for p in self.llm.parameters():
            p.requires_grad = False

        self.runtime_branch_mode = "both"
        self.alignment_losses_enabled = False
        self.loss_w_align = 0.0
        self.loss_w_consistency = 0.0
        self.alignment_temperature = 0.07
        self.alignment_dim = 256
        self.ts_align_head: Optional[nn.Module] = None
        self.vision_align_head: Optional[nn.Module] = None
        self.fused_align_head: Optional[nn.Module] = None
        self.text_align_head: Optional[nn.Module] = None

    @staticmethod
    def _load_llm_with_attn_fallback(
        *,
        llm_id: str,
        device: str,
        llm_attn_impl: str,
    ):
        attn_impl = str(llm_attn_impl).lower()
        if attn_impl not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError(f"Unsupported llm_attn_impl: {llm_attn_impl}")

        candidates = [attn_impl]
        if attn_impl == "flash_attention_2":
            candidates.extend(["sdpa", "eager"])
        elif attn_impl != "eager":
            candidates.append("eager")

        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    llm_id,
                    torch_dtype=torch.bfloat16,
                    device_map={"": device},
                    attn_implementation=candidate,
                    local_files_only=Path(llm_id).exists(),
                )
                if candidate != attn_impl:
                    print(
                        f"⚠️ Failed to load {llm_id} with attn_implementation={attn_impl}; "
                        f"falling back to {candidate}."
                    )
                return model, candidate
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to load {llm_id} with attention implementations {candidates}: {last_error}"
        ) from last_error

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for the LLM to reduce memory usage.
        
        This trades compute for memory by recomputing activations during
        the backward pass instead of storing them.
        """
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            # Non-reentrant checkpointing is more robust with DDP + LoRA training.
            # The default reentrant mode can mark LoRA parameters ready twice during
            # backward when torchrun is used, which breaks phase2 joint training.
            try:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print("✅ Gradient checkpointing enabled for LLM (use_reentrant=False)")
            except TypeError:
                self.llm.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled for LLM")
        else:
            print("⚠️ LLM does not support gradient_checkpointing_enable()")
        if hasattr(self.encoder, "enable_gradient_checkpointing"):
            self.encoder.enable_gradient_checkpointing()

    def forward(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        Forward pass for DDP compatibility.
        
        This method wraps compute_loss to make the model compatible with
        DistributedDataParallel, which requires forward() to be called
        for proper gradient synchronization.
        
        Args:
            batch: List of dictionaries containing the batch data
            
        Returns:
            Loss tensor
        """
        return self.compute_loss(batch)



    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Enable LoRA fine-tuning for the LLM component.

        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of module names to apply LoRA to. If None, uses defaults.
        """
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "peft package is required for LoRA fine-tuning. Please install with: pip install peft"
            )

        if self.lora_enabled:
            raise RuntimeError(
                "LoRA is already enabled. Call disable_lora() first if you want to reconfigure LoRA."
            )

        # Store reference to original model before applying LoRA
        self.original_llm = self.llm

        # Default target modules for common architectures
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

        # Create LoRA config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        try:
            # Apply LoRA to the model
            self.llm = get_peft_model(self.llm, lora_config)
            self.lora_enabled = True

            # Print LoRA info
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
            print(
                "   This might be due to incompatible target modules for your model architecture."
            )
            print(
                "   Try specifying different target_modules or check your model's layer names."
            )
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

    def set_runtime_branch_mode(self, runtime_branch_mode: str):
        runtime_branch_mode = str(runtime_branch_mode).lower()
        if runtime_branch_mode not in {"both", "ts_only", "vision_only"}:
            raise ValueError(f"Unsupported runtime_branch_mode: {runtime_branch_mode}")
        self.runtime_branch_mode = runtime_branch_mode

    def enable_alignment_losses(
        self,
        *,
        loss_w_align: float = 0.2,
        loss_w_consistency: float = 0.1,
        align_dim: int = 256,
        temperature: float = 0.07,
    ):
        if align_dim <= 0:
            raise ValueError("align_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.loss_w_align = float(loss_w_align)
        self.loss_w_consistency = float(loss_w_consistency)
        self.alignment_temperature = float(temperature)
        self.alignment_dim = int(align_dim)

        if self.ts_align_head is None:
            self.ts_align_head = nn.Sequential(
                nn.LayerNorm(ENCODER_OUTPUT_DIM),
                nn.Linear(ENCODER_OUTPUT_DIM, self.alignment_dim),
            ).to(self.device)
        if self.vision_align_head is None:
            self.vision_align_head = nn.Sequential(
                nn.LayerNorm(ENCODER_OUTPUT_DIM),
                nn.Linear(ENCODER_OUTPUT_DIM, self.alignment_dim),
            ).to(self.device)
        if self.fused_align_head is None:
            self.fused_align_head = nn.Sequential(
                nn.LayerNorm(ENCODER_OUTPUT_DIM),
                nn.Linear(ENCODER_OUTPUT_DIM, self.alignment_dim),
            ).to(self.device)
        if self.text_align_head is None:
            self.text_align_head = nn.Sequential(
                nn.LayerNorm(self.llm.config.hidden_size),
                nn.Linear(self.llm.config.hidden_size, self.alignment_dim),
            ).to(self.device)

        self.alignment_losses_enabled = True

    @staticmethod
    def _mean_pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / denom

    def _encode_alignment_texts(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("Alignment texts must not be empty")
        if self.text_align_head is None:
            raise RuntimeError("Alignment losses are not enabled")

        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = tok.input_ids.to(self.device, non_blocking=True)
        attention_mask = tok.attention_mask.to(self.device, non_blocking=True)
        with torch.no_grad():
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]

        pooled = self._mean_pool_hidden_states(hidden_states, attention_mask)
        projected = self.text_align_head(pooled.float())
        return F.normalize(projected, dim=-1)

    def _aggregate_encoder_pooled_outputs(
        self,
        pooled_outputs: Dict[str, Optional[torch.Tensor]],
        ts_counts: List[int],
    ) -> Dict[str, Optional[torch.Tensor]]:
        aggregated: Dict[str, Optional[torch.Tensor]] = {}
        for key, tensor in pooled_outputs.items():
            if tensor is None:
                aggregated[key] = None
                continue

            sample_vectors = []
            offset = 0
            for count in ts_counts:
                if count <= 0:
                    sample_vectors.append(torch.zeros_like(tensor[0]))
                    continue
                sample_vectors.append(tensor[offset : offset + count].mean(dim=0))
                offset += count
            aggregated[key] = torch.stack(sample_vectors, dim=0)
        return aggregated

    def _contrastive_info_nce(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(z_a, z_b.transpose(0, 1)) / self.alignment_temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_a = F.cross_entropy(logits, targets)
        loss_b = F.cross_entropy(logits.transpose(0, 1), targets)
        return 0.5 * (loss_a + loss_b)

    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
        *,
        runtime_branch_mode: Optional[str] = None,
        return_encoder_outputs: bool = False,
    ):
        """
        TL;DR:
            This function is probably the most crucial part of OpenTSLM-SP, and also the hardest to understand.
            It's where the magic happens and legends are made.

            It batches and embeds all text and time series inputs in parallel,
            then reassembles them per sample to allow efficient GPU execution.
            Praise the PyTorch Wizards: ChatGPT-o4-mini-high, Patrick, and Thomas (listed in strictly descending order of skill).

        Long description:
            Processes a batch of training samples by embedding and aligning text and time series data
            for efficient parallel processing on the GPU.

            This method performs the following steps:

            1. Extracts all text components (pre_prompt, time_series_text, post_prompt) from each sample,
            and embeds them in a single batch using the LLM tokenizer and embedding layer. Padding and attention
            masks are applied to accommodate variable-length sequences.

            2. Gathers all time series segments across the batch and pads them
            into a single tensor of shape [N_ts_total, T_padded, D], where T_padded
            is the smallest multiple of `patch_size` ≥ the longest segment length.
            This tensor is then encoded and projected into the LLM hidden space.

            3. After all embeddings are extracted, the function reconstructs each original sample by interleaving its
            embedded pre_prompt, time series texts and corresponding time series embeddings, and the post_prompt, preserving original order.

            4. Pads all reassembled sequences to a uniform length across the batch to form the final input tensor
                and attention mask.

            5. All of this is only required for efficient processing.

        - pre_prompt: str
        - time_series_text: List[str]
        - time_series: Tensor [N_ts, T] or [N_ts, T, D]
        - post_prompt: str
        Returns (inputs_embeds, attention_mask)
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
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [N_all, P_max, H]

        # 3) Batch time-series encode & project
        ts_list: List[torch.Tensor] = []
        for sample in batch:
            for ts in sample["time_series"]:
                # ensure [T] → [T,1]
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                ts_list.append(ts)

        encoder_outputs = None
        if ts_list:
            ts_padded = pad_sequence(ts_list, batch_first=True).to(
                device, non_blocking=True
            )
            # ── pad time dim to multiple of patch_size ──
            T_max = ts_padded.size(1)
            rem = T_max % self.patch_size
            if rem:
                pad_len = self.patch_size - rem
                pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
                ts_padded = torch.cat([ts_padded, pad], dim=1)
            # ── now ts_padded: [N_ts_total, T_padded, 1]

            # ── key fix: squeeze out the feature dim so encoder sees [B, L] ──
            if self.encoder_type == "newts_dual_branch":
                ts_enc = self.encoder(
                    ts_padded.squeeze(-1),
                    runtime_branch_mode=runtime_branch_mode or self.runtime_branch_mode,
                    return_intermediates=return_encoder_outputs,
                )
            else:
                ts_enc = self.encoder(ts_padded.squeeze(-1))

            if return_encoder_outputs and self.encoder_type == "newts_dual_branch":
                encoder_outputs = ts_enc
                ts_enc = encoder_outputs["fused_tokens"]
            ts_proj = self.projector(ts_enc).to(
                text_embeds.dtype
            )  # [N_ts_total, N_patches, H]
        else:
            ts_proj = torch.empty(0, 0, H, device=device, dtype=text_embeds.dtype)

        # 4) Re‐assemble per sample
        all_seq_embeds, all_seq_masks = [], []
        ts_offset = 0
        for (start, end), n_ts in zip(text_ptrs, ts_counts):
            sample_embeds = text_embeds[start:end]  # [1+N_ts+1, P_max, H]
            sample_masks = attn_mask[start:end]  # [1+N_ts+1, P_max]
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

                proj = ts_proj[ts_offset + i]  # [N_patches, H]
                seq_embeds.append(proj)
                seq_masks.append(
                    torch.ones(proj.size(0), device=device, dtype=torch.long)
                )

            ts_offset += n_ts

            # post_prompt (fixed)
            length = sample_masks[-1].sum().item()
            seq_embeds.append(sample_embeds[-1, :length, :])
            seq_masks.append(sample_masks[-1, :length])

            all_seq_embeds.append(torch.cat(seq_embeds, dim=0))
            all_seq_masks.append(torch.cat(seq_masks, dim=0))

        # 5) Batch-pad the final sequences
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)  # [B, L_max, H]
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)  # [B, L_max]

        if not return_encoder_outputs:
            return inputs_embeds, attention_mask

        pooled_outputs = self._aggregate_encoder_pooled_outputs(
            {
                "pooled_ts": encoder_outputs.get("pooled_ts") if encoder_outputs is not None else None,
                "pooled_vision": encoder_outputs.get("pooled_vision") if encoder_outputs is not None else None,
                "pooled_fused": encoder_outputs.get("pooled_fused") if encoder_outputs is not None else None,
            },
            ts_counts=ts_counts,
        )
        return inputs_embeds, attention_mask, {
            **pooled_outputs,
            "effective_branch_mode": (
                encoder_outputs.get("effective_branch_mode") if encoder_outputs is not None else None
            ),
        }

    def generate(
        self,
        batch: List[Dict[str, any]],
        max_new_tokens: int = 50,
        runtime_branch_mode: Optional[str] = None,
        skip_special_tokens: bool = True,
        **generate_kwargs,
    ) -> List[str]:
        inputs_embeds, attention_mask = self.pad_and_apply_batch(
            batch,
            runtime_branch_mode=runtime_branch_mode,
        )
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=skip_special_tokens)

    def compute_losses(
        self,
        batch: List[Dict[str, any]],
        *,
        runtime_branch_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        batch: same format as generate()
        answers: List[str] of length B
        """
        answers = [b["answer"] for b in batch]
        sample_level_answer_loss = any(
            sample.get("answer_loss_normalization") == "sample" for sample in batch
        )

        encoder_outputs = None
        batch_outputs = self.pad_and_apply_batch(
            batch,
            runtime_branch_mode=runtime_branch_mode,
            return_encoder_outputs=self.alignment_losses_enabled,
        )
        if self.alignment_losses_enabled:
            inputs_embeds, attention_mask, encoder_outputs = batch_outputs
        else:
            inputs_embeds, attention_mask = batch_outputs
        B, L, H = inputs_embeds.size()

        # tokenize answers,但不添加bos token
        ans_tok = self.tokenizer(
            answers, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
        )
        ans_ids = ans_tok.input_ids.to(self.device, non_blocking=True)
        ans_mask = ans_tok.attention_mask.to(self.device, non_blocking=True)
        ans_emb = self.llm.get_input_embeddings()(ans_ids)  # [B, A_max, H]

        # append
        inputs_embeds = torch.cat([inputs_embeds, ans_emb], dim=1)  # [B, L+A, H]
        attention_mask = torch.cat([attention_mask, ans_mask], dim=1)  # [B, L+A]

        # labels: only on the answer tokens
        total_len = attention_mask.size(1)
        labels = torch.full((B, total_len), -100, device=self.device, dtype=torch.long)
        if sample_level_answer_loss:
            labels[:, L:] = torch.where(
                ans_mask.bool(),
                ans_ids,
                torch.full_like(ans_ids, -100),
            )
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid_mask = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid_mask, 0)
            token_losses = F.cross_entropy(
                shift_logits.float().reshape(-1, shift_logits.size(-1)),
                safe_labels.reshape(-1),
                reduction="none",
            ).view(B, -1)
            token_counts = valid_mask.sum(dim=1).clamp_min(1)
            loss_lm = ((token_losses * valid_mask).sum(dim=1) / token_counts).mean()
        else:
            labels[:, L:] = ans_ids
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss_lm = outputs.loss
        loss_align = torch.zeros((), device=self.device, dtype=loss_lm.dtype)
        loss_consistency = torch.zeros((), device=self.device, dtype=loss_lm.dtype)
        loss_total = loss_lm

        if self.alignment_losses_enabled and encoder_outputs is not None:
            pooled_fused = encoder_outputs.get("pooled_fused")
            if pooled_fused is not None and self.fused_align_head is not None:
                align_indices = [
                    idx
                    for idx, sample in enumerate(batch)
                    if sample.get("alignment_target_text")
                ]
                if align_indices:
                    z_fused = self.fused_align_head(pooled_fused[align_indices].float())
                    z_fused = F.normalize(z_fused, dim=-1)
                    z_text = self._encode_alignment_texts(
                        [batch[idx]["alignment_target_text"] for idx in align_indices]
                    )
                    loss_align = self._contrastive_info_nce(z_fused, z_text).to(loss_lm.dtype)

            pooled_ts = encoder_outputs.get("pooled_ts")
            pooled_vision = encoder_outputs.get("pooled_vision")
            if (
                pooled_ts is not None
                and pooled_vision is not None
                and self.ts_align_head is not None
                and self.vision_align_head is not None
            ):
                z_ts = F.normalize(self.ts_align_head(pooled_ts.float()), dim=-1)
                z_vi = F.normalize(self.vision_align_head(pooled_vision.float()), dim=-1)
                loss_consistency = (1.0 - F.cosine_similarity(z_ts, z_vi, dim=-1)).mean().to(loss_lm.dtype)

            loss_total = loss_total + self.loss_w_align * loss_align + self.loss_w_consistency * loss_consistency

        return {
            "loss_total": loss_total,
            "loss_lm": loss_lm,
            "loss_align": loss_align,
            "loss_consistency": loss_consistency,
        }

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        return self.compute_losses(batch)["loss_total"]

    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "llm_id": self.llm_id,
            "encoder_type": self.encoder_type,
            "encoder_config": self.encoder_config,
        }

    def store_to_file(self, path: str):
        checkpoint = {
            "model_config": self.get_checkpoint_metadata(),
            "encoder_state": self.encoder.state_dict(),
            "projector_state": self.projector.state_dict(),
        }

        # Add LoRA state to checkpoint
        self.save_lora_state_to_checkpoint(checkpoint)

        torch.save(checkpoint, path)

    def load_from_file(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])

        # Load LoRA state if present (allow missing for backward compatibility)
        self.load_lora_state_from_checkpoint(ckpt, allow_missing=True)

        print(f"📥 Loaded model from epoch {ckpt.get('epoch', '?')}")

    def load_lora_state_from_checkpoint(
        self, checkpoint: dict, allow_missing: bool = False
    ):
        """
        Load LoRA adapters from a checkpoint.

        Args:
            checkpoint: Checkpoint dictionary containing potential LoRA state
            allow_missing: If True, don't raise exception when checkpoint has no LoRA but model expects it

        Raises:
            RuntimeError: When there's a mismatch between checkpoint and current LoRA state
        """
        checkpoint_has_lora = checkpoint.get("lora_enabled", False)

        if checkpoint_has_lora and "lora_state" in checkpoint:
            # Checkpoint has LoRA adapters
            if not self.lora_enabled:
                raise RuntimeError(
                    "Checkpoint contains LoRA adapters but LoRA is not currently enabled. "
                    "Call enable_lora() before loading this checkpoint."
                )

            # Load LoRA adapters
            try:
                lora_state = checkpoint["lora_state"]
                loaded_count = 0
                missing_keys = []

                # Track which LoRA parameters we expect to find
                expected_lora_params = {
                    name
                    for name, param in self.llm.named_parameters()
                    if "lora_" in name
                }

                for name, param in self.llm.named_parameters():
                    if name in lora_state and "lora_" in name:
                        param.data.copy_(lora_state[name])
                        loaded_count += 1
                    elif "lora_" in name:
                        missing_keys.append(name)

                if missing_keys and not allow_missing:
                    raise RuntimeError(
                        f"Could not find LoRA parameters in checkpoint: {missing_keys[:5]}..."
                    )

                print(f"📥 Loaded LoRA adapters: {loaded_count} parameters")
                return loaded_count

            except Exception as e:
                if "Could not find LoRA parameters" in str(e):
                    raise  # Re-raise our custom exception
                raise RuntimeError(f"Failed to load LoRA adapters: {e}")

        elif checkpoint_has_lora:
            if not allow_missing:
                raise RuntimeError(
                    "Checkpoint indicates LoRA was enabled but no LoRA state found"
                )
            print("⚠️  Checkpoint indicates LoRA was enabled but no LoRA state was found.")
            print("   LoRA adapters will keep their current initialization.")
            return 0

        # Handle case where checkpoint has no LoRA but model expects it
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

    def save_lora_state_to_checkpoint(self, checkpoint: dict):
        """
        Save LoRA adapters to a checkpoint dictionary.

        Args:
            checkpoint: Checkpoint dictionary to add LoRA state to

        Returns:
            int: Number of LoRA parameters saved
        """
        checkpoint["lora_enabled"] = self.lora_enabled

        if self.lora_enabled and hasattr(self.llm, "peft_config"):
            try:
                # Save LoRA adapter weights
                lora_state = {}
                for name, param in self.llm.named_parameters():
                    if "lora_" in name:
                        lora_state[name] = param.data.clone()

                if lora_state:
                    checkpoint["lora_state"] = lora_state
                    checkpoint["lora_config"] = self.llm.peft_config
                    print(f"💾 Saved LoRA adapters with {len(lora_state)} parameters")
                    return len(lora_state)
            except Exception as e:
                raise RuntimeError(f"Failed to save LoRA adapters: {e}")

        return 0

    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 30000, normalize: bool = False
    ) -> str:
        """
        Evaluate a prompt and return the generated text.
        """

        batch = [prompt.to_dict()]
        self.eval()
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            batch,
            patch_size=self.patch_size,
            normalize=normalize,
        )
        output = self.generate(batch, max_new_tokens=max_new_tokens)
        return output[0]

    @staticmethod
    def _build_default_tslanet_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        default_config = {
            "output_dim": ENCODER_OUTPUT_DIM,
            "patch_size": 8,
            "emb_dim": 128,
            "depth": 2,
            "dropout": 0.15,
        }
        if config:
            default_config.update(config)
        return default_config

    @staticmethod
    def _build_default_newts_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        default_config = {
            "output_dim": ENCODER_OUTPUT_DIM,
            "patch_length": 16,
            "stride": 8,
            "d_model": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 3,
            "ffn_dim": 512,
            "dropout": 0.1,
            "dynamic_length": True,
            "ts_positional_encoding": "sinusoidal",
            "branch_mode": "both",
            "vit_model_name": "facebook/dinov2-base",
            "vit_feature_mode": "single",
            "vit_layer_idx": 4,
            "vit_mix_layers": None,
            "vit_patch_size": 16,
            "vit_stride": 0.5,
            "vision_2d_mode": "legacy_unfold",
            "vit_truncate_to_feature_layer": True,
            "vit_num_hidden_layers": None,
            "projector_type": "mlp",
            "projector_dropout": 0.1,
            "use_pma": False,
            "aggregator_layers": 2,
            "aggregator_hidden_size": ENCODER_OUTPUT_DIM,
            "aggregator_num_heads": 8,
            "aggregator_ffn_dim": ENCODER_OUTPUT_DIM * 4,
            "aggregator_num_queries": 4,
            "aggregator_query_mode": "shared",
            "aggregator_fusion_mode": "gated_sum",
            "aggregator_gate_type": "dynamic",
            "aggregator_fuse_layers": 1,
            "enable_modality_embeddings": False,
            "branch_dropout": 0.0,
            "vision_train_mode": "none",
            "vision_topk_blocks": 4,
            "freeze_ts_backbone": False,
            "freeze_vision_backbone": True,
        }
        if config:
            default_config.update(config)
        return default_config

    def _build_encoder(
        self,
        *,
        encoder_type: str,
        device: str,
        encoder_pretrained_path: Optional[str],
        tslanet_config: Optional[Dict[str, Any]],
        newts_dual_branch_config: Optional[Dict[str, Any]],
    ):
        if encoder_type == "transformer_cnn":
            encoder = TransformerCNNEncoder().to(device)
            return encoder, encoder.patch_size, {}

        if encoder_type == "tslanet":
            from ..encoder.TSLANetEncoder import TSLANetEncoder

            config = self._build_default_tslanet_config(tslanet_config)
            encoder = TSLANetEncoder(**config).to(device)
            if encoder_pretrained_path:
                encoder.load_pretrained(encoder_pretrained_path)
                print(f"✅ Loaded TSLANet pretrained weights from: {encoder_pretrained_path}")
            return encoder, config.get("patch_size", 8), config

        if encoder_type == "newts_dual_branch":
            config = self._build_default_newts_config(newts_dual_branch_config)
            encoder = NewTSDualBranchEncoder(**config, device=device).to(device)
            return encoder, 1, encoder.get_config()

        raise ValueError(f"Unsupported encoder_type: {encoder_type}")
