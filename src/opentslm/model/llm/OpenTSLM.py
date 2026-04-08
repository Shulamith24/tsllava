# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
import torch
from typing import Any, Dict, Optional, TYPE_CHECKING, Union
from enum import Enum
from huggingface_hub import hf_hub_download

from .OpenTSLMSP import OpenTSLMSP

if TYPE_CHECKING:
    from .OpenTSLMFlamingo import OpenTSLMFlamingo


class ModelType(Enum):
    """Enumeration of supported model types."""

    SP = "sp"
    FLAMINGO = "flamingo"


class OpenTSLM:
    """
    Factory class for loading EmbedHealth models from Hugging Face Hub.

    Automatically detects model type based on repository ID suffix and returns
    the appropriate model instance (EmbedHealthSP or EmbedHealthFlamingo) with
    optimal parameters from curriculum learning training.

    - Repository IDs ending with "-sp" load EmbedHealthSP models
    - Repository IDs ending with "-flamingo" load EmbedHealthFlamingo models

    The factory automatically applies the exact same parameters used in curriculum learning:
    - EmbedHealthSP: Uses default constructor parameters
    - EmbedHealthFlamingo: cross_attn_every_n_layers=1, gradient_checkpointing=False

    These parameters are fixed and cannot be overridden since they were determined during training.

    Example:
        >>> model = OpenTSLM.load_pretrained("OpenTSLM/gemma-3-270m-pt-sleep-flamingo")
        >>>
        >>> from opentslm.prompt.full_prompt import FullPrompt
        >>> prompt = FullPrompt(...)
        >>> response = model.eval_prompt(prompt)
    """

    @classmethod
    def load_pretrained(
        cls,
        repo_id: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_lora: Optional[bool] = False,
        checkpoint_path: Optional[str] = None,
        llm_attn_impl: str = "sdpa",
    ) -> Union[OpenTSLMSP, "OpenTSLMFlamingo"]:
        """
        Load a pretrained model from Hugging Face Hub.

        Args:
            repo_id: Hugging Face repository ID (e.g., "OpenTSLM/gemma-3-270m-pt-sleep-flamingo")
            device: Device to load the model on (default: auto-detect)
            cache_dir: Directory to cache downloaded models (optional)
            enable_lora: Whether to enable LoRA (default: False)
            checkpoint_path: Optional local path to a previously downloaded checkpoint file

        Returns:
            Union[OpenTSLMSP, OpenTSLMFlamingo]: The loaded model instance

        Example:
            >>> model = OpenTSLM.load_pretrained("OpenTSLM/gemma-3-270m-pt-sleep-flamingo")
            >>> prompt = FullPrompt(...)
            >>> response = model.eval_prompt(prompt)
        """
        device = cls._get_device(device)
        model_type = cls._detect_model_type(repo_id)
        checkpoint_path = checkpoint_path or cls._download_model_files(repo_id, cache_dir)
        base_llm_id = cls._get_base_llm_id(repo_id)
        checkpoint = None
        resolved_llm_id = base_llm_id
        resolved_encoder_type = None
        if model_type == ModelType.SP:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            sp_init_kwargs = cls._resolve_sp_init_kwargs_from_checkpoint(
                checkpoint=checkpoint,
                fallback_llm_id=base_llm_id,
            )
            resolved_llm_id = sp_init_kwargs["llm_id"]
            resolved_encoder_type = sp_init_kwargs["encoder_type"]

        print(f"🚀 Loading {model_type.value.upper()} model...")
        print(f"   Repository: {repo_id}")
        print(f"   Base LLM: {resolved_llm_id}")
        if resolved_encoder_type is not None:
            print(f"   Encoder: {resolved_encoder_type}")
        print(f"   Device: {device}")

        if model_type == ModelType.SP:
            model = OpenTSLMSP(
                llm_id=sp_init_kwargs["llm_id"],
                device=device,
                encoder_type=sp_init_kwargs["encoder_type"],
                tslanet_config=sp_init_kwargs["tslanet_config"],
                newts_dual_branch_config=sp_init_kwargs["newts_dual_branch_config"],
                llm_attn_impl=llm_attn_impl,
            )
            if enable_lora:
                lora_r, lora_alpha = cls._resolve_lora_hparams_from_checkpoint(checkpoint)
                model.enable_lora(lora_r=lora_r, lora_alpha=lora_alpha)
        elif model_type == ModelType.FLAMINGO:
            from .OpenTSLMFlamingo import OpenTSLMFlamingo

            # OpenTSLMFlamingo with fixed parameters from curriculum learning
            model = OpenTSLMFlamingo(
                device=device,
                llm_id=base_llm_id,
                cross_attn_every_n_layers=1,
                gradient_checkpointing=False,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load the checkpoint
        model.load_from_file(checkpoint_path)
        model.eval()

        print(f"✅ {model_type.value.upper()} model loaded successfully!")
        return model

    @staticmethod
    def _resolve_sp_init_kwargs_from_checkpoint(
        checkpoint: Dict[str, Any],
        fallback_llm_id: str,
    ) -> Dict[str, Any]:
        model_config = checkpoint.get("model_config") or {}
        encoder_type = model_config.get("encoder_type") or OpenTSLM._infer_sp_encoder_type_from_checkpoint(checkpoint)
        encoder_config = model_config.get("encoder_config") or {}

        if not encoder_config and encoder_type == "tslanet":
            encoder_config = OpenTSLM._infer_legacy_tslanet_config(checkpoint)

        return {
            "llm_id": model_config.get("llm_id") or fallback_llm_id,
            "encoder_type": encoder_type,
            "tslanet_config": dict(encoder_config) if encoder_type == "tslanet" else None,
            "newts_dual_branch_config": (
                dict(encoder_config) if encoder_type == "newts_dual_branch" else None
            ),
        }

    @staticmethod
    def _resolve_lora_hparams_from_checkpoint(checkpoint: Dict[str, Any]) -> tuple[int, int]:
        lora_config = checkpoint.get("lora_config")
        config_obj: Any = lora_config

        if isinstance(lora_config, dict):
            config_obj = lora_config.get("default")
            if config_obj is None and lora_config:
                config_obj = next(iter(lora_config.values()))

        lora_r = getattr(config_obj, "r", None)
        lora_alpha = getattr(config_obj, "lora_alpha", None)

        if isinstance(config_obj, dict):
            lora_r = config_obj.get("r", lora_r)
            lora_alpha = config_obj.get("lora_alpha", lora_alpha)

        return int(lora_r) if lora_r is not None else 16, int(lora_alpha) if lora_alpha is not None else 32

    @staticmethod
    def _infer_sp_encoder_type_from_checkpoint(checkpoint: Dict[str, Any]) -> str:
        encoder_state = checkpoint.get("encoder_state") or {}
        if not encoder_state:
            return "transformer_cnn"

        keys = list(encoder_state.keys())
        if any(key.startswith("ts_backbone.") or key.startswith("vision_encoder.") or key.startswith("aggregator.") for key in keys):
            return "newts_dual_branch"
        if any(key.startswith("tsla_blocks.") or key.startswith("patch_embed.proj.") for key in keys):
            return "tslanet"
        return "transformer_cnn"

    @staticmethod
    def _infer_legacy_tslanet_config(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        encoder_state = checkpoint.get("encoder_state") or {}
        patch_weight = encoder_state.get("patch_embed.proj.weight")
        pos_embed = encoder_state.get("pos_embed")

        depth = 0
        for key in encoder_state:
            if key.startswith("tsla_blocks."):
                try:
                    depth = max(depth, int(key.split(".")[1]) + 1)
                except (IndexError, ValueError):
                    continue

        config: Dict[str, Any] = {}
        if patch_weight is not None:
            config["patch_size"] = int(patch_weight.shape[-1])
            config["emb_dim"] = int(patch_weight.shape[0])
        if depth > 0:
            config["depth"] = depth
        if pos_embed is not None and "emb_dim" not in config:
            config["emb_dim"] = int(pos_embed.shape[-1])
        return config

    @staticmethod
    def _get_device(device: Optional[str]) -> str:
        """Auto-detect device if not specified."""
        if device is not None:
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def _detect_model_type(repo_id: str) -> ModelType:
        """Detect model type from repository ID suffix."""
        if repo_id.endswith("-sp"):
            return ModelType.SP
        elif repo_id.endswith("-flamingo"):
            return ModelType.FLAMINGO
        else:
            raise ValueError(
                f"Repository ID '{repo_id}' must end with either '-sp' or '-flamingo' "
                f"to indicate the model type."
            )

    @staticmethod
    def _download_model_files(repo_id: str, cache_dir: Optional[str] = None) -> str:
        """Download model checkpoint from Hugging Face Hub."""
        try:
            # Download the main model checkpoint file
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename="model_checkpoint.pt",
                cache_dir=cache_dir,
                local_files_only=False,
            )
            print(f"✅ Downloaded model checkpoint from {repo_id}")
            return checkpoint_path

        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from {repo_id}. "
                f"Tried 'model_checkpoint.pt'. "
                f"Original error: {e}"
            )

    @staticmethod
    def _get_base_llm_id(repo_id: str) -> str:
        """Get the base LLM ID from static mapping based on repository ID pattern."""
        repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

        # Extract base model from repository name pattern
        if repo_name.startswith("llama-3.2-3b"):
            return "meta-llama/Llama-3.2-3B"
        elif repo_name.startswith("llama-3.2-1b"):
            return "meta-llama/Llama-3.2-1B"
        elif repo_name.startswith("gemma-3-1b"):
            return "google/gemma-3-1b"
        elif repo_name.startswith("gemma-3-270m"):
            return "google/gemma-3-270m"
        else:
            # Raise exception if pattern doesn't match
            raise ValueError(
                f"Unable to determine base LLM ID from repository name '{repo_name}'. "
                f"Repository name must start with one of: 'llama-3.2-3b', 'llama-3.2-1b', "
                f"'gemma-3-1b', or 'gemma-3-270m'."
            )
