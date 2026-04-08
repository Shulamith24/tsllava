# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
from typing import Any, Dict, Optional, Sequence

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


LEGACY_VISION_2D_MODE = "legacy_unfold"
TIVIT_SQRT_OVERLAP_VISION_2D_MODE = "tivit_sqrt_overlap"
SUPPORTED_VISION_2D_MODES = (
    LEGACY_VISION_2D_MODE,
    TIVIT_SQRT_OVERLAP_VISION_2D_MODE,
)
LEGACY_DEFAULT_VIT_STRIDE = 0.5
TIVIT_DEFAULT_VIT_STRIDE = 0.1
DEPRECATED_VISION_2D_MODE_HINTS = {
    "adaptive_unfold": "Use 'tivit_sqrt_overlap' instead.",
    "reshape_serpentine": "Choose 'legacy_unfold' or 'tivit_sqrt_overlap' explicitly.",
}


def validate_vision_2d_mode(vision_2d_mode: str) -> str:
    if vision_2d_mode in SUPPORTED_VISION_2D_MODES:
        return vision_2d_mode
    if vision_2d_mode in DEPRECATED_VISION_2D_MODE_HINTS:
        hint = DEPRECATED_VISION_2D_MODE_HINTS[vision_2d_mode]
        raise ValueError(f"vision_2d_mode '{vision_2d_mode}' has been removed. {hint}")
    raise ValueError(
        f"Unsupported vision_2d_mode: {vision_2d_mode}. "
        f"Valid modes: {list(SUPPORTED_VISION_2D_MODES)}"
    )


def resolve_effective_vision_stride(
    vision_2d_mode: str,
    ts_stride: float,
    *,
    stride_explicit: bool,
) -> float:
    resolved_mode = validate_vision_2d_mode(vision_2d_mode)
    if stride_explicit:
        return float(ts_stride)
    if resolved_mode == TIVIT_SQRT_OVERLAP_VISION_2D_MODE:
        return TIVIT_DEFAULT_VIT_STRIDE
    return LEGACY_DEFAULT_VIT_STRIDE


def resolve_effective_vit_patch_policy(vision_2d_mode: str) -> str:
    resolved_mode = validate_vision_2d_mode(vision_2d_mode)
    if resolved_mode == TIVIT_SQRT_OVERLAP_VISION_2D_MODE:
        return "sqrt_time_length"
    return "fixed"


def vision_mode_ignores_patch_size(vision_2d_mode: str) -> bool:
    return validate_vision_2d_mode(vision_2d_mode) == TIVIT_SQRT_OVERLAP_VISION_2D_MODE


def load_vision_backbone(
    model_name: str,
    *,
    num_hidden_layers: Optional[int] = None,
):
    """
    Load a vision backbone and its paired image processor.

    When ``num_hidden_layers`` is provided, only the prefix of encoder blocks is
    instantiated during ``from_pretrained``. This reduces memory compared to
    loading the full backbone and discarding layers afterwards.
    """
    model_name_lower = model_name.lower()
    model_kwargs = {}
    if num_hidden_layers is not None:
        model_kwargs["num_hidden_layers"] = num_hidden_layers

    if "dinov2" in model_name_lower:
        from transformers import AutoImageProcessor, Dinov2Model

        processor = AutoImageProcessor.from_pretrained(model_name)
        vit = Dinov2Model.from_pretrained(model_name, **model_kwargs)
        hidden_dim = vit.config.hidden_size
        image_size = vit.config.image_size
        patch_size = vit.config.patch_size
        num_patches = (image_size // patch_size) ** 2
        num_layers = len(vit.encoder.layer)
    elif "clip" in model_name_lower:
        from transformers import CLIPImageProcessor, CLIPVisionModel

        processor = CLIPImageProcessor.from_pretrained(model_name)
        vit = CLIPVisionModel.from_pretrained(model_name, **model_kwargs)
        hidden_dim = vit.config.hidden_size
        image_size = vit.config.image_size
        patch_size = vit.config.patch_size
        num_patches = (image_size // patch_size) ** 2
        num_layers = len(vit.vision_model.encoder.layers)
    elif "siglip" in model_name_lower:
        from transformers import AutoProcessor, SiglipVisionModel

        processor = AutoProcessor.from_pretrained(model_name)
        vit = SiglipVisionModel.from_pretrained(model_name, **model_kwargs)
        hidden_dim = vit.config.hidden_size
        image_size = vit.config.image_size
        patch_size = vit.config.patch_size
        num_patches = (image_size // patch_size) ** 2
        num_layers = len(vit.vision_model.encoder.layers)
    elif "mae" in model_name_lower:
        from transformers import AutoImageProcessor, ViTMAEModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        vit = ViTMAEModel.from_pretrained(model_name, **model_kwargs)
        hidden_dim = vit.config.hidden_size
        image_size = vit.config.image_size
        patch_size = vit.config.patch_size
        num_patches = (image_size // patch_size) ** 2
        num_layers = len(vit.encoder.layer)
    else:
        raise ValueError(f"Unsupported vision model: {model_name}")

    return processor, vit, hidden_dim, num_patches, image_size, num_layers


class NewTSPseudoImageTransform:
    """
    Standalone 1D->2D time-series image transform used by the NewTS vision branch.

    This helper is intentionally lightweight so visualization scripts can reuse
    the exact same pseudo-image construction without loading a vision backbone.
    """

    def __init__(
        self,
        *,
        ts_patch_size: int = 16,
        ts_stride: float = 0.5,
        vision_2d_mode: str = LEGACY_VISION_2D_MODE,
        image_size: int = 224,
    ):
        self.ts_patch_size = int(ts_patch_size)
        self.ts_stride = float(ts_stride)
        self.vision_2d_mode = validate_vision_2d_mode(vision_2d_mode)
        self.image_size = int(image_size)

        if self.ts_patch_size <= 0:
            raise ValueError("ts_patch_size must be positive")
        if not (0.0 < self.ts_stride <= 1.0):
            raise ValueError(f"ts_stride must be in (0, 1], got {self.ts_stride}")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

    @staticmethod
    def _normalize_time_series(x: torch.Tensor) -> torch.Tensor:
        median = x.median(1, keepdim=True)[0]
        q_tensor = torch.tensor([0.75, 0.25], device=x.device, dtype=x.dtype)
        q75, q25 = torch.quantile(x, q_tensor, dim=1, keepdim=True)
        x = x - median
        return x / ((q75 - q25) + 1e-5)

    @staticmethod
    def _normalize_image_grid(x_2d: torch.Tensor) -> torch.Tensor:
        min_vals = x_2d.amin(dim=(-2, -1), keepdim=True)
        max_vals = x_2d.amax(dim=(-2, -1), keepdim=True)
        x_2d = (x_2d - min_vals) / (max_vals - min_vals + 1e-5)
        return torch.pow(x_2d, 0.8)

    @staticmethod
    def _replicate_pad_right(x: torch.Tensor, pad_right: int) -> torch.Tensor:
        if pad_right <= 0:
            return x
        last_value = x[:, -1:].expand(-1, pad_right)
        return torch.cat([x, last_value], dim=1)

    @staticmethod
    def _replicate_pad_left(x: torch.Tensor, pad_left: int) -> torch.Tensor:
        if pad_left <= 0:
            return x
        first_value = x[:, :1].expand(-1, pad_left)
        return torch.cat([first_value, x], dim=1)

    @staticmethod
    def _square_pad(x_2d: torch.Tensor) -> torch.Tensor:
        height = x_2d.size(-2)
        width = x_2d.size(-1)
        if height == width:
            return x_2d

        if height < width:
            delta = width - height
            pad_top = delta // 2
            pad_bottom = delta - pad_top
            return F.pad(x_2d, (0, 0, pad_top, pad_bottom), mode="replicate")

        delta = height - width
        pad_left = delta // 2
        pad_right = delta - pad_left
        return F.pad(x_2d, (pad_left, pad_right, 0, 0), mode="replicate")

    def _resize_square_grid(self, x_2d: torch.Tensor) -> torch.Tensor:
        x_square = self._square_pad(x_2d)
        current_size = x_square.size(-1)
        if current_size == self.image_size:
            return x_square

        resize_mode = "nearest" if current_size < self.image_size else "area"
        return F.interpolate(
            x_square,
            size=(self.image_size, self.image_size),
            mode=resize_mode,
        )

    @staticmethod
    def _compute_unfold_pad_right(
        time_length: int,
        patch_size: int,
        stride_length: int,
    ) -> int:
        if stride_length == patch_size:
            return (patch_size - time_length % patch_size) % patch_size
        if time_length < patch_size:
            return patch_size - time_length
        return (stride_length - (time_length - patch_size) % stride_length) % stride_length

    def _build_unfold_grid(
        self,
        x: torch.Tensor,
        *,
        patch_size: int,
        stride: float,
    ) -> torch.Tensor:
        stride_length = patch_size if stride == 1.0 else max(1, int(round(patch_size * stride)))
        pad_right = self._compute_unfold_pad_right(x.size(1), patch_size, stride_length)
        x = self._replicate_pad_right(x, pad_right)
        if stride_length == patch_size:
            return einops.rearrange(x, "n (h w) -> n 1 h w", w=patch_size)

        x_2d = x.unfold(dimension=1, size=patch_size, step=stride_length)
        return einops.rearrange(x_2d, "n h w -> n 1 h w")

    @staticmethod
    def _compute_tivit_pad_left(
        time_length: int,
        patch_size: int,
        stride_length: int,
    ) -> int:
        if stride_length == patch_size:
            return (patch_size - time_length % patch_size) % patch_size
        if time_length < patch_size:
            return patch_size - time_length
        remainder = (time_length - patch_size) % stride_length
        return stride_length - remainder if remainder != 0 else 0

    def _build_tivit_grid(
        self,
        x: torch.Tensor,
        *,
        patch_size: int,
        stride: float,
    ) -> torch.Tensor:
        stride_length = patch_size if stride == 1.0 else max(1, int(patch_size * stride))
        pad_left = self._compute_tivit_pad_left(x.size(1), patch_size, stride_length)
        x = self._replicate_pad_left(x, pad_left)
        if stride_length == patch_size:
            return einops.rearrange(x, "n (p f) -> n 1 f p", f=patch_size)

        x_2d = x.unfold(dimension=1, size=patch_size, step=stride_length)
        return einops.rearrange(x_2d, "n h w -> n 1 h w")

    def _resize_tivit_grid(self, x_2d: torch.Tensor) -> torch.Tensor:
        current_height = x_2d.size(-2)
        current_width = x_2d.size(-1)
        if current_height == self.image_size and current_width == self.image_size:
            return x_2d
        return F.interpolate(
            x_2d,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )

    def get_runtime_transform_config(
        self,
        *,
        time_length: int,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> Dict[str, Any]:
        requested_patch_size = int(patch_size or self.ts_patch_size)
        requested_stride = float(stride or self.ts_stride)
        if requested_patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if not (0.0 < requested_stride <= 1.0):
            raise ValueError(f"stride must be in (0, 1], got {requested_stride}")

        if self.vision_2d_mode == TIVIT_SQRT_OVERLAP_VISION_2D_MODE:
            effective_patch_size = max(1, int(math.sqrt(time_length)))
            effective_stride_length = (
                effective_patch_size
                if requested_stride == 1.0
                else max(1, int(effective_patch_size * requested_stride))
            )
        else:
            effective_patch_size = requested_patch_size
            effective_stride_length = (
                effective_patch_size
                if requested_stride == 1.0
                else max(1, int(round(effective_patch_size * requested_stride)))
            )

        return {
            "vision_2d_mode": self.vision_2d_mode,
            "requested_patch_size": requested_patch_size,
            "requested_stride_ratio": requested_stride,
            "effective_patch_size": effective_patch_size,
            "effective_stride_ratio": requested_stride,
            "effective_stride_length": effective_stride_length,
            "effective_vit_patch_policy": resolve_effective_vit_patch_policy(self.vision_2d_mode),
        }

    def ts2grid(
        self,
        x: torch.Tensor,
        *,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> torch.Tensor:
        x = self._normalize_time_series(x)
        x = einops.rearrange(x, "b t d -> (b d) t")
        time_length = x.shape[-1]
        runtime_config = self.get_runtime_transform_config(
            time_length=time_length,
            patch_size=patch_size,
            stride=stride,
        )

        if self.vision_2d_mode == TIVIT_SQRT_OVERLAP_VISION_2D_MODE:
            x_2d = self._build_tivit_grid(
                x,
                patch_size=runtime_config["effective_patch_size"],
                stride=runtime_config["effective_stride_ratio"],
            )
        else:
            x_2d = self._build_unfold_grid(
                x,
                patch_size=runtime_config["effective_patch_size"],
                stride=runtime_config["effective_stride_ratio"],
            )

        return self._normalize_image_grid(x_2d)

    def ts2grayscale_image(
        self,
        x: torch.Tensor,
        *,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> torch.Tensor:
        x_grid = self.ts2grid(x, patch_size=patch_size, stride=stride)
        if self.vision_2d_mode == TIVIT_SQRT_OVERLAP_VISION_2D_MODE:
            return self._resize_tivit_grid(x_grid)
        return self._resize_square_grid(x_grid)

    def ts2image(
        self,
        x: torch.Tensor,
        *,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> torch.Tensor:
        x_resized = self.ts2grayscale_image(x, patch_size=patch_size, stride=stride)
        return einops.repeat(x_resized, "b 1 h w -> b c h w", c=3)


class NewTSVisionEncoder(nn.Module):
    """
    TiViT-style time-series-to-image encoder with optional feature-layer truncation.

    Feature layers follow the original 1-based indexing used in ``temp/newts``:
    ``layer_idx=4`` means the output after encoder block 4.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        layer_idx: int = 4,
        feature_mode: str = "single",
        mix_layers: Optional[Sequence[int]] = None,
        ts_patch_size: int = 16,
        ts_stride: float = 0.5,
        vision_2d_mode: str = LEGACY_VISION_2D_MODE,
        image_size: Optional[int] = None,
        return_cls_token: bool = False,
        truncate_to_feature_layer: bool = True,
        num_hidden_layers: Optional[int] = None,
        vision_train_mode: str = "none",
        vision_topk_blocks: int = 4,
        device: str = "cuda",
    ):
        super().__init__()

        self.model_name = model_name
        self.layer_idx = int(layer_idx)
        self.feature_mode = feature_mode
        self.mix_layers = tuple(int(layer) for layer in mix_layers) if mix_layers else tuple()
        self.ts_patch_size = int(ts_patch_size)
        self.ts_stride = float(ts_stride)
        self.vision_2d_mode = validate_vision_2d_mode(vision_2d_mode)
        self.return_cls_token = return_cls_token
        self.truncate_to_feature_layer = bool(truncate_to_feature_layer)
        self.requested_num_hidden_layers = self._resolve_requested_num_hidden_layers(
            num_hidden_layers=num_hidden_layers
        )
        self.vision_train_mode = str(vision_train_mode)
        self.vision_topk_blocks = int(vision_topk_blocks)
        self.device = device

        (
            self.processor,
            self.vit,
            self.hidden_dim,
            self.num_vit_patches,
            backbone_image_size,
            self.num_hidden_layers,
        ) = load_vision_backbone(
            model_name,
            num_hidden_layers=self.requested_num_hidden_layers,
        )

        self.image_size = int(image_size or backbone_image_size)
        self.mix_logits: Optional[nn.Parameter] = None
        self._validate_feature_config()

    def _resolve_requested_num_hidden_layers(
        self,
        *,
        num_hidden_layers: Optional[int],
    ) -> Optional[int]:
        if num_hidden_layers is not None and num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive when provided")

        if self.feature_mode == "single":
            if self.layer_idx <= 0:
                raise ValueError("single feature_mode requires a positive 1-based layer_idx")
            target_depth = self.layer_idx
        elif self.feature_mode == "scalar_mix":
            layers = self.mix_layers or (4, 8, 12)
            target_depth = max(int(layer) for layer in layers)
        elif self.feature_mode == "last":
            target_depth = None
        else:
            raise ValueError(
                f"feature_mode must be one of ['last', 'single', 'scalar_mix'], got {self.feature_mode}"
            )

        if num_hidden_layers is not None and target_depth is not None and num_hidden_layers < target_depth:
            raise ValueError(
                "num_hidden_layers must be greater than or equal to the requested feature layer depth"
            )

        if num_hidden_layers is not None:
            return int(num_hidden_layers)

        if not self.truncate_to_feature_layer:
            return None

        return target_depth

    def _validate_layer_index(self, layer: int) -> int:
        if layer < 1:
            raise ValueError(f"Layer index must be a positive 1-based integer, got {layer}")
        if layer > self.num_hidden_layers:
            raise ValueError(
                f"Layer index {layer} exceeds loaded backbone depth {self.num_hidden_layers}"
            )
        return layer

    def _validate_feature_config(self):
        if self.feature_mode == "single":
            self.layer_idx = self._validate_layer_index(self.layer_idx)
            self.mix_layers = tuple()
            self.mix_logits = None
            return

        if self.feature_mode == "scalar_mix":
            layers = self.mix_layers or (4, 8, 12)
            deduped_layers = []
            for layer in layers:
                layer_idx = self._validate_layer_index(int(layer))
                if layer_idx not in deduped_layers:
                    deduped_layers.append(layer_idx)
            self.mix_layers = tuple(deduped_layers)
            self.mix_logits = nn.Parameter(torch.zeros(len(self.mix_layers), dtype=torch.float32))
            self.layer_idx = -1
            return

        self.layer_idx = -1
        self.mix_layers = tuple()
        self.mix_logits = None

    @staticmethod
    def _to_pil_image(image: torch.Tensor) -> Image.Image:
        image = image.detach().cpu().clamp(0.0, 1.0)
        image = (image * 255.0).round().to(torch.uint8)
        image_np = image.permute(1, 2, 0).numpy()
        return Image.fromarray(image_np)

    def _make_image_transform(self) -> NewTSPseudoImageTransform:
        return NewTSPseudoImageTransform(
            ts_patch_size=self.ts_patch_size,
            ts_stride=self.ts_stride,
            vision_2d_mode=self.vision_2d_mode,
            image_size=self.image_size,
        )

    def ts2grid(
        self,
        x: torch.Tensor,
        *,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> torch.Tensor:
        return self._make_image_transform().ts2grid(
            x,
            patch_size=patch_size,
            stride=stride,
        )

    def ts2grayscale_image(
        self,
        x: torch.Tensor,
        *,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> torch.Tensor:
        return self._make_image_transform().ts2grayscale_image(
            x,
            patch_size=patch_size,
            stride=stride,
        )

    def ts2image(
        self,
        x: torch.Tensor,
        *,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> torch.Tensor:
        return self._make_image_transform().ts2image(
            x,
            patch_size=patch_size,
            stride=stride,
        )

    def get_runtime_transform_config(
        self,
        *,
        time_length: int,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._make_image_transform().get_runtime_transform_config(
            time_length=time_length,
            patch_size=patch_size,
            stride=stride,
        )

    def _prepare_pixel_values(self, images: torch.Tensor) -> torch.Tensor:
        image_list = [self._to_pil_image(image) for image in images]
        inputs = self.processor(images=image_list, return_tensors="pt")
        return inputs["pixel_values"].to(images.device)

    def forward_vit(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_pixel_values(images)
        outputs = self.vit(
            pixel_values=pixel_values,
            output_hidden_states=(self.feature_mode in {"single", "scalar_mix"}),
        )

        if self.feature_mode == "last":
            token_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("hidden_states were not returned by the vision backbone")

            if self.feature_mode == "single":
                token_states = hidden_states[self.layer_idx]
            else:
                layer_states = [hidden_states[layer] for layer in self.mix_layers]
                stacked_states = torch.stack(layer_states, dim=0)
                alpha = torch.softmax(self.mix_logits, dim=0).to(
                    device=stacked_states.device,
                    dtype=stacked_states.dtype,
                )
                token_states = torch.einsum("l,lbph->bph", alpha, stacked_states)

        if self.return_cls_token:
            return token_states
        return token_states[:, 1:, :]

    def forward(
        self,
        past_values: torch.Tensor,
        n_vars: int = 1,
    ) -> torch.Tensor:
        num_vars = int(n_vars or past_values.size(-1))
        if num_vars == 1:
            images = self.ts2image(past_values)
            return self.forward_vit(images)

        feature_blocks = []
        for var_idx in range(num_vars):
            images = self.ts2image(past_values[:, :, var_idx : var_idx + 1])
            feature_blocks.append(self.forward_vit(images))
        return torch.cat(feature_blocks, dim=-1)

    def freeze(self):
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.vit.parameters():
            param.requires_grad = True

    def _get_embeddings_module(self) -> Optional[nn.Module]:
        if hasattr(self.vit, "embeddings"):
            return self.vit.embeddings
        if hasattr(self.vit, "vision_model") and hasattr(self.vit.vision_model, "embeddings"):
            return self.vit.vision_model.embeddings
        return None

    def _get_encoder_blocks(self):
        if hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
            return list(self.vit.encoder.layer)
        if hasattr(self.vit, "vision_model") and hasattr(self.vit.vision_model, "encoder"):
            encoder = self.vit.vision_model.encoder
            if hasattr(encoder, "layers"):
                return list(encoder.layers)
        return []

    def _get_final_norm_modules(self):
        modules = []
        for attr_name in ("layernorm", "post_layernorm", "post_layernorm", "norm", "pre_layrnorm"):
            module = getattr(self.vit, attr_name, None)
            if isinstance(module, nn.Module):
                modules.append(module)
        if hasattr(self.vit, "vision_model"):
            for attr_name in ("post_layernorm", "pre_layrnorm", "layernorm"):
                module = getattr(self.vit.vision_model, attr_name, None)
                if isinstance(module, nn.Module):
                    modules.append(module)
        return modules

    @staticmethod
    def _set_module_requires_grad(module: Optional[nn.Module], requires_grad: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def set_trainable_blocks(
        self,
        mode: str = "none",
        topk: Optional[int] = None,
    ) -> Dict[str, Any]:
        mode = str(mode).lower()
        if mode not in {"none", "topk", "all"}:
            raise ValueError(f"Unsupported vision_train_mode: {mode}")

        if topk is None:
            topk = self.vision_topk_blocks
        topk = int(topk)
        if topk < 0:
            raise ValueError("vision_topk_blocks must be >= 0")

        self.freeze()
        blocks = self._get_encoder_blocks()
        trainable_indices = []

        if mode == "all":
            self.unfreeze()
            trainable_indices = list(range(len(blocks)))
        elif mode == "topk":
            if blocks:
                self._set_module_requires_grad(self._get_embeddings_module(), True)
                for module in self._get_final_norm_modules():
                    self._set_module_requires_grad(module, True)
                if topk > 0:
                    start_idx = max(0, len(blocks) - topk)
                    trainable_indices = list(range(start_idx, len(blocks)))
                    for idx in trainable_indices:
                        self._set_module_requires_grad(blocks[idx], True)

        self.vision_train_mode = mode
        self.vision_topk_blocks = topk
        return {
            "train_mode": self.vision_train_mode,
            "loaded_layers": len(blocks),
            "trainable_layers": trainable_indices,
        }

    def enable_gradient_checkpointing(self):
        if hasattr(self.vit, "gradient_checkpointing_enable"):
            self.vit.gradient_checkpointing_enable()

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def get_num_patches(self) -> int:
        return self.num_vit_patches

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def get_feature_config(self) -> Dict[str, Any]:
        return {
            "feature_mode": self.feature_mode,
            "layer_idx": self.layer_idx if self.feature_mode == "single" else None,
            "mix_layers": list(self.mix_layers) if self.feature_mode == "scalar_mix" else None,
            "vision_2d_mode": self.vision_2d_mode,
            "truncate_to_feature_layer": self.truncate_to_feature_layer,
            "num_hidden_layers": self.requested_num_hidden_layers,
            "loaded_num_hidden_layers": self.num_hidden_layers,
            "vision_train_mode": self.vision_train_mode,
            "vision_topk_blocks": self.vision_topk_blocks,
        }

    def get_learned_mix_alpha(self) -> Optional[Dict[str, Any]]:
        if self.mix_logits is None:
            return None
        alpha = torch.softmax(self.mix_logits.detach().cpu(), dim=0)
        return {
            "layers": list(self.mix_layers),
            "alpha": [float(value) for value in alpha.tolist()],
        }
