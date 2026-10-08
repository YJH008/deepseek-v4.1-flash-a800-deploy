#!/usr/bin/env bash
# DeepSeek-V4.1-Flash 部署（4 × A800-SXM4-80GB，GPU 0-3）
# 适配自 shi3z/deepseekv4.1-A100-custom 仓库
# 适用场景：单流高质量回答 / GPU 故障降级（如 GPU6 损坏，隔离后仅用 0-3）
set -euo pipefail

# ---- 隔离故障卡：让 torch 只能看到物理卡 0,1,2,3 ----
# 关键：GPU 故障时 torch 在 import 阶段会枚举全部卡做 capability 检查导致崩溃，
#       用 CUDA_VISIBLE_DEVICES 让 torch 根本看不到故障卡即可绕过。
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---- 路径 ----
REPO_DIR="/share/model/deepseekv4.1-A100-custom"
CKPT="/share/model/DeepSeek-V4.1-Flash"
CONDA_BIN="/root/miniconda3/envs/dsv41/bin"
PYTHON="${CONDA_BIN}/python"
cd "${REPO_DIR}"

# ---- CUDA 内核编译：仓库硬编码了 /usr/local/cuda-12.8，机器上是 12.0 ----
export DSV41_NVCC="/usr/local/cuda-12.0/bin/nvcc"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 分段预填充 ----
export DSV41_MOE_PREFILL_CHUNK=2048
export DSV41_ENGRAM_PREFILL_CHUNK=512
export DSV41_HC_PREFILL_CHUNK=256
export DSV41_SPARSE_ATTN_CHUNK=128

# ---- 4 卡：40 层均分（每卡 10 层）、384 experts 均分（每卡 96）----
export DSV41_LAYER_COUNTS=10,10,10,10
export DSV41_EP_COMPACT_XQ=1
export DSV41_EP_GRAPH_TOKENS=1048576
export DSV41_EP_CAND_TOKENS=1048576

# ---- 动态缓存（初始 32K，按需增长到 1M）----
export DSV41_CACHE_INIT_TOKENS=32768
export DSV41_EP_PREALLOC_TOKENS=32768
export DSV41_EXACT_CACHE_GROW=1

# ---- 加速 ----
export DSV41_CED=1
export DSV41_GPU_SLOT_CACHE=1
export DSV41_PREFIX_DEDUP_MIRRORS=1

# ---- 并发：单流高质量（2 解码槽 + 1 预填 scratchpad = max-seqs 3）----
export DSV41_MAX_SEQS=3

# ---- 采样惩罚 ----
export DSV41_REPETITION_PENALTY=1.05
export DSV41_FREQUENCY_PENALTY=0.02
export DSV41_PRESENCE_PENALTY=0.0
export DSV41_PENALTY_WINDOW=2048
export DSV41_BAN_CYCLES=1
export DSV41_LOOP_DETECT=1

# ---- 前缀缓存（tmpfs）----
export DSV41_PREFIX_CACHE_DIR=/dev/shm/dsv41-prefix-cache
export DSV41_PREFIX_CACHE_ENTRIES=16
export DSV41_PREFIX_CACHE_GB=32
export DSV41_PREFIX_TMPFS_ENTRIES=16
export DSV41_PREFIX_TMPFS_GB=32
export DSV41_PREFIX_BLOCK_REPLAY=1
export DSV41_PREFIX_BLOCK_SIZE=512
mkdir -p /dev/shm/dsv41-prefix-cache

exec "${PYTHON}" -u -m dsv41.serve \
    --ckpt "${CKPT}" \
    --devices 0,1,2,3 \
    --ep \
    --ep-shards 96,96,96,96 \
    --max-seq-len 1048576 \
    --max-seqs 3 \
    --host 0.0.0.0 \
    --port 8000 \
    --mtp 0