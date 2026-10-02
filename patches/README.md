# 8 卡适配补丁

原作者仓库只支持 4 卡（GPU 0-3）。要在 8×A800 上运行，需打以下 3 个补丁。

## 补丁 1：`dsv41/serve.py`（必需）

**位置**：第 795 行（`main()` 函数内）

**原代码**：
```python
dev_list = [int(d) for d in a.devices.split(",") if int(d) in (0, 1, 2, 3)]
```

**改为**：
```python
dev_list = [int(d) for d in a.devices.split(",") if d.strip() != ""]
```

**原因**：原作者硬编码了设备白名单 `(0,1,2,3)`，任何 id ≥ 4 的 GPU 都会被过滤掉，导致 8 卡参数被解析成只有 4 卡，报错
`ValueError: DSV41_LAYER_COUNTS must be 4 positive counts summing to 40`。

## 补丁 2：`dsv41/stats.py`（可选，影响监控显示）

**位置**：第 29 行

**原代码**：
```python
# Strict constraint: only GPUs 0, 1, 2, 3 are permitted for DeepSeek inference
PERMITTED_GPUS = (0, 1, 2, 3)
```

**改为**：
```python
# GPUs permitted for DeepSeek inference (expanded to 0..7 for 8-GPU A800 deployment)
PERMITTED_GPUS = (0, 1, 2, 3, 4, 5, 6, 7)
```

**原因**：不修改则 dashboard 监控只显示 GPU 0-3，不影响推理功能，但监控不完整。

## 补丁 3：CUDA 内核编译路径（必需，环境变量）

仓库 `dsv41/cukern.py` 第 13 行硬编码了 nvcc 路径：

```python
NVCC = os.environ.get("DSV41_NVCC", "/usr/local/cuda-12.8/bin/nvcc")
```

若机器 CUDA 版本不是 12.8，加载权重时编译 `.cu` 内核会失败（找不到 nvcc）。通过环境变量覆盖：

```bash
export DSV41_NVCC=/usr/local/cuda-12.0/bin/nvcc
```

本仓库的 `run_8xa800.sh` 已内置该环境变量。

---

## 快速 patch 命令

```bash
cd <repo>/dsv41

# 补丁 1：serve.py
sed -i 's/if int(d) in (0, 1, 2, 3)]/if d.strip() != ""]/' serve.py

# 补丁 2：stats.py
sed -i 's/PERMITTED_GPUS = (0, 1, 2, 3)/PERMITTED_GPUS = (0, 1, 2, 3, 4, 5, 6, 7)/' stats.py
```

> 建议手动核对后再执行，确保 sed 匹配准确。