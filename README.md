# DeepSeek-V4.1-Flash 在 A800 上的部署方案（8 卡 / 4 卡）

> 基于 [shi3z/deepseekv4.1-A100-custom](https://github.com/shi3z/deepseekv4.1-A100-custom) 的自定义推理运行时，在 **NVIDIA A800-SXM4-80GB** 上完成 DeepSeek-V4.1-Flash 的实盘部署与适配。支持 **8 卡**与 **4 卡**两种模式，提供一键切换方法。

## 一、部署结果

| 项目 | 结果 |
|---|---|
| GPU | 8 × A800-SXM4-80GB（compute_cap 8.0 = sm80） |
| 模型 | DeepSeek-V4.1-Flash（40 层 / 384 experts MoE，FP8 dense + FP4 experts） |
| 上下文 | 最高 1M token |
| 单流解码 | ~38–50 tok/s（实测） |
| API | OpenAI 兼容 `http://<host>:8000/v1` |
| 验证 | `/health` → `{"status":"ok"}`，`/v1/chat/completions` 正常返回 |

## 二、硬件 / 环境要求

| 项目 | 要求 | 本机实测 |
|---|---|---|
| GPU | A100 / A800（sm80），80GB 显存 | 8 × A800-SXM4-80GB ✅ |
| 驱动 | 支持 CUDA 12.x | 575.57.08 ✅ |
| CUDA Toolkit | 11.8+ / 12.x | 12.0（`/usr/local/cuda-12.0`） |
| 主机内存 | 数十~上百 GB（Engram 表 91.6 GiB/层） | 1 TB ✅ |
| 磁盘 | 权重约 476 GB | `/share` 8.2T（gpfs）✅ |
| Python | 3.8+（推荐 3.11） | 3.11.15（conda） |
| OS | 任意 Linux | Rocky Linux 8.6 |

## 三、CPU / GPU 数量适配

原作者仓库的启动脚本与源码**硬编码了 4 卡（GPU 0-3）**。要在 8 卡上运行，必须打下面三个补丁，否则：

- `--devices 0,1,2,3,4,5,6,7` 会被过滤成只剩 `0,1,2,3`，报错 `DSV41_LAYER_COUNTS must be 4 positive counts`。

三个补丁见 [patches/](./patches/README.md)：

1. `dsv41/serve.py` 第 795 行：去掉 `if int(d) in (0,1,2,3)` 的设备白名单过滤，改为保留所有非空设备 id。
2. `dsv41/stats.py` 第 29 行：`PERMITTED_GPUS` 从 `(0,1,2,3)` 扩到 `(0..7)`（否则 dashboard 只显示 4 卡）。
3. 环境变量 `DSV41_NVCC=/usr/local/cuda-12.0/bin/nvcc`（仓库硬编码了 `cuda-12.8` 路径）。

## 四、部署步骤

### 1. 准备权重

本方案直接使用官方发布的 HF 格式权重（扁平化 key 命名，与该运行时完全兼容），无需二次转换。

```
/share/model/DeepSeek-V4.1-Flash/
├── model-00001-of-00048.safetensors   # …共 48 个分片（476G）
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── config.json
└── inference/config.json              # load.py 会读这个路径
```

> 注意：`dsv41/load.py` 读取的是 `{ckpt}/inference/config.json`（第 250 行），确认该文件存在。

### 2. 建 conda 环境并安装依赖

```bash
conda create -n dsv41 python=3.11 -y
conda activate dsv41

# 关键：torch 必须用 cu124 版本（驱动 575.x 不支持 cu130）
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install triton==3.2.0
pip install transformers sympy numpy psutil safetensors Pillow
```

> **踩坑**：默认 `pip install torch` 会装 `2.14.0+cu130`，因驱动过旧报 `driver too old`，`torch.cuda.is_available()` 为 False。必须改成 `2.6.0+cu124`。

### 3. 拉取源码并打 8 卡补丁

```bash
git clone https://github.com/shi3z/deepseekv4.1-A100-custom.git
cd deepseekv4.1-A100-custom
# 按 patches/README.md 打 3 个补丁
```

### 4. 启动服务

根据卡数选择对应脚本（详见下文 [六、4 卡 / 8 卡切换](#六4-卡--8-卡切换)）。

**8 卡模式**（[run_8xa800.sh](./run_8xa800.sh)）：

```bash
bash run_8xa800.sh
# 或后台托管：tmux new-session -d -s dsv41 ./start_tmux.sh
```

核心参数（8 卡 EP 分配）：

```
--devices 0,1,2,3,4,5,6,7
--ep --ep-shards 48,48,48,48,48,48,48,48   # 384 experts ÷ 8
DSV41_LAYER_COUNTS=5,5,5,5,5,5,5,5          # 40 层 ÷ 8
--max-seq-len 1048576 --max-seqs 3 --mtp 0
```

**4 卡模式**（[run_4xa800.sh](./run_4xa800.sh)）：

```bash
bash run_4xa800.sh
# 或后台托管：tmux new-session -d -s dsv41 ./start_tmux.sh
```

核心参数（4 卡 EP 分配，含故障卡隔离）：

```
export CUDA_VISIBLE_DEVICES=0,1,2,3        # 隔离故障卡，torch 只看到 4 卡
--devices 0,1,2,3
--ep --ep-shards 96,96,96,96               # 384 experts ÷ 4
DSV41_LAYER_COUNTS=10,10,10,10             # 40 层 ÷ 4
--max-seq-len 1048576 --max-seqs 3 --mtp 0
```

### 5. 验证

```bash
curl http://127.0.0.1:8000/health
# → {"status": "ok"}

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":200,"temperature":0.6}'
```

## 五、关键踩坑记录

| 问题 | 现象 | 解决 |
|---|---|---|
| torch 版本 | 默认 cu130，`driver too old` | `torch==2.6.0+cu124` |
| triton 版本 | 与 torch 不匹配 | `triton==3.2.0` |
| 8 卡被硬编码成 4 卡 | `DSV41_LAYER_COUNTS must be 4` | 打 serve.py 补丁 |
| CUDA 编译路径 | 硬编码 cuda-12.8 | `DSV41_NVCC` 环境变量覆盖 |
| Pillow 缺失 | `No module named 'PIL'` | `pip install Pillow` |
| SSH 后台进程被杀 | paramiko/SSH 会话关闭后服务停止 | 用 tmux 托管（弃用 systemd，因 restart 反复打断权重加载） |
| GPU 故障干扰 torch 初始化 | torch 在 `import` 时枚举全部卡做 capability 检查，故障卡导致 CUDA 上下文崩溃（`device=N, num_gpus=<乱码>`） | `CUDA_VISIBLE_DEVICES` 隔离故障卡，详见下文 |
| tmux 会话没起来 | `tmux kill-session` 只杀会话，tmux server 残留导致新会话失效 | 用 `tmux kill-server` 彻底清理 |
| SSE 流式不关闭连接 | `[DONE]` 后 curl 不自动断开、Cherry Studio 卡等待 | 详见 [patches/fix-sse-close.md](./patches/fix-sse-close.md) |

## 六、4 卡 / 8 卡切换

### 6.1 两种模式对比

| 维度 | 8 卡 | 4 卡 |
|---|---|---|
| 每卡层数 | 5 层 | 10 层 |
| 每卡 experts | 48 | 96 |
| 单卡显存占用 | ~37-38 GiB | ~72-75 GiB |
| 显存余量 | 充足 | 较紧（约占 90%） |
| 适用场景 | 默认均衡生产、大神存余量 | 单流高质量、**GPU 故障降级** |

### 6.2 切换方法

切换本质是改 `start_tmux.sh` 里调用的脚本（`run_8xa800.sh` ↔ `run_4xa800.sh`），然后重启服务。

**方法一：改 start_tmux.sh（推荐）**

编辑 `start_tmux.sh`，把最后一行改为目标脚本：

```bash
# 切 4 卡
bash run_4xa800.sh 2>&1 | tee -a serve.log

# 切 8 卡
bash run_8xa800.sh 2>&1 | tee -a serve.log
```

然后重启：

```bash
tmux kill-server          # 彻底清理（注意不是 kill-session）
sleep 2
tmux new-session -d -s dsv41 <repo>/start_tmux.sh
```

**方法二：直接指定脚本前台运行**

```bash
cd <repo>
bash run_4xa800.sh   # 或 run_8xa800.sh
```

### 6.3 GPU 故障降级（重要）

当某张 GPU 故障（如 GPU6 掉卡）时，即使只传 `--devices 0,1,2,3`，torch 仍会在 `import` 阶段枚举**全部物理卡**做 capability 检查，导致整个 CUDA 上下文崩溃。

**解决**：使用 `CUDA_VISIBLE_DEVICES` 隔离，让 torch 根本看不到故障卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 只暴露健康的 4 张卡
```

> `run_4xa800.sh` 已内置该隔离。注意 `CUDA_VISIBLE_DEVICES` 会把可见卡重新编号为 0,1,2,3，后续 `--devices 0,1,2,3` 参数无需改动。

**故障卡如何判断**：

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv,noheader
# 故障卡 utilization 显示 [N/A]，且 nvidia-smi -i <id> -q 直接超时
dmesg | grep -i nvrm | tail   # 出现 NV_ERR_GPU_IN_FULLCHIP_RESET / nvlink status 失败
```

> `CUDA_VISIBLE_DEVICES` 只是软件隔离，卡本身仍是坏的。若 `nvidia-smi` 持续报 FULLCHIP_RESET / NVLink 错误，建议**联系硬件运维更换故障卡**。

## 七、已知限制（来自上游）

- **非真正 token 级流式**：`eng.generate_text()` 一次性生成完整回答后才返回，SSE 是"伪流式"（先算完再分块吐出）。接入 Cherry Studio 等聊天客户端时，建议**关闭客户端的流式输出开关**，否则长回答生成期间客户端会一直"等待"。
- 无连续批处理（仅槽位级并发）。
- 无视觉输入（尽管仓库含 vision 代码，README 仍标注限制）。
- 长上下文（1M）需 4 卡流水线并行 + 特定环境变量。

## 八、常用管理命令

```bash
# 看日志
tail -f <repo>/serve.log

# 停服务（kill-server 彻底清理 tmux server，避免残留）
tmux kill-server

# 重新拉起（服务器重启后 / 切换卡数后）
tmux new-session -d -s dsv41 <repo>/start_tmux.sh
```

## 九、文件说明

| 文件 | 说明 |
|---|---|
| `README.md` | 本文档 |
| `run_8xa800.sh` | 8 卡单流高质量启动脚本 |
| `run_4xa800.sh` | 4 卡启动脚本（含 `CUDA_VISIBLE_DEVICES` 故障卡隔离） |
| `start_tmux.sh` | tmux 托管启动包装（决定切 4 卡或 8 卡的入口） |
| `patches/README.md` | 8 卡适配补丁说明 |

## 免责声明

本方案仅供学习与技术交流，权重文件请从官方合法渠道获取。部署和使用大模型请遵守相关法律法规与模型许可证。