# 修复：SSE 流式响应结束后连接不关闭（curl / Cherry Studio 卡等待）

## 现象

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

`data: [DONE]` 已输出，但 curl **不自动断开**，需要手动 `Ctrl+C`。Cherry Studio 等聊天客户端表现为「一句话结束后一直转圈等待」。

## 根因

`dsv41/serve.py` 使用 Python 标准库 `BaseHTTPRequestHandler`，且：

```python
protocol_version = "HTTP/1.1"   # 第 75 行，默认 keep-alive 持久连接
```

SSE 流式响应分支存在两个 bug：

1. **硬编码 `Connection: keep-alive`**（原第 245、592 行）
2. **写完 `data: [DONE]` 后直接 `return`**（原第 429、699 行），既不关闭连接，也不发 `Content-Length`

SSE 响应没有 `Content-Length`（流式长度未知），客户端只能依靠「连接关闭（EOF）」判断流结束。但服务端写完 `[DONE]` 后既不关连接、也不发长度，于是客户端永远等不到 EOF，一直挂起。

> 对比：非流式路径（`_json` 方法）正确发了 `Content-Length`，所以非流式请求能正常结束。

## 修复

共 4 处改动：

1. 两处 `Connection: keep-alive` → `Connection: close`（第 245、592 行）
2. 四处 `data: [DONE]` 写入后追加 `self.close_connection = True`（含异常兜底分支）

```python
# 改动 1（两处）
self.send_header("Connection", "close")

# 改动 2（四处，紧跟每个 [DONE] 写入之后）
_sse_write(b"data: [DONE]\n\n")
self.close_connection = True  # SSE 流结束，主动关闭连接触发客户端 EOF
```

## 验证

修复后：

```
data: [DONE]

[curl结束] http=200 total=0.47s   ← 自动断开，退出码 0（不再挂起）
```

## 快速 patch 命令

```bash
cd <repo>/dsv41

# 改动 1：Connection keep-alive -> close
sed -i 's/self.send_header("Connection", "keep-alive")/self.send_header("Connection", "close")/g' serve.py

# 改动 2：每个 _sse_write(b"data: [DONE]\n\n") 后加 close_connection
# （建议用脚本逐行插入，见下）
```

改动 2 的脚本化插入逻辑（Python）：

```python
lines = open('serve.py').read().split('\n')
out = []
for ln in lines:
    out.append(ln)
    if ln.strip() == '_sse_write(b"data: [DONE]\\n\\n")':
        indent = ln[:len(ln)-len(ln.lstrip())]
        out.append(indent + 'self.close_connection = True  # SSE 流结束，主动关闭连接')
open('serve.py','w').write('\n'.join(out))
```

> 注意：改完用 `python -m py_compile serve.py` 验证语法，再重启服务。