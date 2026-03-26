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
        vision_2d_mode: str = "legacy_unfold",
        image_size: Optional[int] = None,
        return_cls_token: bool = False,
        truncate_to_feature_layer: bool = True,
        num_hidden_layers: Optional[int] = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.model_name = model_name
        self.layer_idx = int(layer_idx)
        self.feature_mode = feature_mode
        self.mix_layers = tuple(int(layer) for layer in mix_layers) if mix_layers else tuple()
        self.ts_patch_size = int(ts_patch_size)
        self.ts_stride = float(ts_stride)
        self.vision_2d_mode = vision_2d_mode
        self.return_cls_token = return_cls_token
        self.truncate_to_feature_layer = bool(truncate_to_feature_layer)
        self.requested_num_hidden_layers = self._resolve_requested_num_hidden_layers(
            num_hidden_layers=num_hidden_layers
        )
        self.device = device

        if self.vision_2d_mode not in {"legacy_unfold", "adaptive_unfold", "reshape_serpentine"}:
            raise ValueError(f"Unsupported vision_2d_mode: {self.vision_2d_mode}")

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
    def _candidate_search_bounds(time_length: int) -> tuple[int, int]:
        min_allowed = 1 if time_length < 4 else 4
        base = max(1, int(round(math.sqrt(time_length))))
        lower = max(min_allowed, base - 4)
        upper = min(time_length, base + 4)
        if lower > upper:
            lower = upper = min(time_length, max(1, base))
        return lower, upper

    def _select_serpentine_layout(self, time_length: int) -> tuple[int, int]:
        lower, upper = self._candidate_search_bounds(time_length)
        best_width = lower
        best_height = math.ceil(time_length / lower)
        best_score = (abs(best_height - lower), best_height * lower - time_length, -lower)

        for width in range(lower, upper + 1):
            height = math.ceil(time_length / width)
            score = (abs(height - width), height * width - time_length, -width)
            if score < best_score:
                best_width = width
                best_height = height
                best_score = score

        return best_height, best_width

    def _build_serpentine_grid(self, x: torch.Tensor) -> torch.Tensor:
        time_length = x.size(1)
        height, width = self._select_serpentine_layout(time_length)
        total_cells = height * width
        x = self._replicate_pad_right(x, total_cells - time_length)
        x_2d = x.reshape(x.size(0), 1, height, width)
        if height > 1:
            x_2d[:, :, 1::2, :] = torch.flip(x_2d[:, :, 1::2, :], dims=(-1,))
        return x_2d

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

    def _select_adaptive_patch_size(
        self,
        time_length: int,
        *,
        stride: float,
    ) -> int:
        stride_scale = stride if stride < 1.0 else 1.0
        target_patch = max(1, int(round(math.sqrt(time_length / stride_scale))))
        min_allowed = 1 if time_length < 4 else 4
        lower = max(min_allowed, target_patch - 4)
        upper = min(time_length, target_patch + 4)
        if lower > upper:
            lower = upper = min(time_length, max(1, target_patch))

        best_patch = lower
        stride_length = lower if stride == 1.0 else max(1, int(round(lower * stride)))
        pad_right = self._compute_unfold_pad_right(time_length, lower, stride_length)
        num_rows = ((time_length + pad_right - lower) // stride_length) + 1
        best_score = (abs(num_rows - lower), pad_right, -lower)

        for patch_size in range(lower, upper + 1):
            stride_length = patch_size if stride == 1.0 else max(1, int(round(patch_size * stride)))
            pad_right = self._compute_unfold_pad_right(time_length, patch_size, stride_length)
            num_rows = ((time_length + pad_right - patch_size) // stride_length) + 1
            score = (abs(num_rows - patch_size), pad_right, -patch_size)
            if score < best_score:
                best_patch = patch_size
                best_score = score

        return best_patch

    def ts2image(
        self,
        x: torch.Tensor,
        *,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> torch.Tensor:
        patch_size = int(patch_size or self.ts_patch_size)
        stride = float(stride or self.ts_stride)
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if not (0.0 < stride <= 1.0):
            raise ValueError(f"stride must be in (0, 1], got {stride}")
        x = self._normalize_time_series(x)
        x = einops.rearrange(x, "b t d -> (b d) t")
        time_length = x.shape[-1]

        if self.vision_2d_mode == "reshape_serpentine":
            x_2d = self._build_serpentine_grid(x)
        elif self.vision_2d_mode == "adaptive_unfold":
            adaptive_patch_size = self._select_adaptive_patch_size(time_length, stride=stride)
            x_2d = self._build_unfold_grid(x, patch_size=adaptive_patch_size, stride=stride)
        else:
            x_2d = self._build_unfold_grid(x, patch_size=patch_size, stride=stride)

        x_2d = self._normalize_image_grid(x_2d)
        x_resized = self._resize_square_grid(x_2d)
        return einops.repeat(x_resized, "b 1 h w -> b c h w", c=3)

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
        }

    def get_learned_mix_alpha(self) -> Optional[Dict[str, Any]]:
        if self.mix_logits is None:
            return None
        alpha = torch.softmax(self.mix_logits.detach().cpu(), dim=0)
        return {
            "layers": list(self.mix_layers),
            "alpha": [float(value) for value in alpha.tolist()],
        }
