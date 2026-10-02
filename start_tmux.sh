#!/bin/bash
# tmux 托管启动包装脚本
# 用法：tmux new-session -d -s dsv41 /path/to/start_tmux.sh
export PATH=/root/miniconda3/envs/dsv41:$PATH
export DSV41_NVCC=/usr/local/cuda-12.0/bin/nvcc
cd "$(dirname "$0")"
bash run_8xa800.sh 2>&1 | tee -a serve.log