# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

# ---------------------------
# Hyper-parameters for LLM Classification
# ---------------------------

BATCH_SIZE = 4

PATCH_SIZE = 4  # 时序编码器 patch 大小
NUM_EPOCHS = 30
EARLY_STOP_PAT = 10
LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4
LR_LORA = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
WARMUP_FRAC = 0.03
MAX_SAMPLES = None

# 编码器配置
EMBED_DIM = 128
ENCODER_OUTPUT_DIM = EMBED_DIM
TRANSFORMER_INPUT_DIM = EMBED_DIM
