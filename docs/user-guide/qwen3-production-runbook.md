# QuantInfer Qwen3 量化与推理生产 Runbook

本文档适用于 Qwen3 Dense、CPU Scalar、单序列 Prefill/Decode 和 Q8_0 线性层量化。当前仓库提供 CLI 推理入口，`quantinfer-server` 不提供 HTTP 服务。

生产代码固定为经过 CI 验证的 commit。本文示例使用 `f6a44e9fd265c5f06bf323cef02a2230a4ec5221`，发布时替换为实际批准的 commit。

## 1. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| 模型 | Qwen3 Dense |
| 设备 | CPU Scalar |
| 输入权重 | 单个 SafeTensors 文件 |
| GGUF 浮点类型 | F32、F16、BF16，加载时转为 F32 |
| 量化 | Q8_0，量化线性投影层 |
| 生成 | 单序列 Prefill、Decode、KV Cache、EOS、temperature、seed |
| tokenizer | tokenizer.json 的 BPE vocab 和 merges |
| 服务 | CLI 单次生成 |

Qwen3 MoE、sliding-window、custom RoPE scaling、GPU 后端、分片 SafeTensors、HTTP 服务和多请求调度不在当前支持范围内。

## 2. 主机准备

以下命令适用于 Debian/Ubuntu。部署目录由 `quantinfer` 用户管理。

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ca-certificates coreutils python3-full python3-venv cargo rustc
sudo useradd --system --home /opt/quantinfer --shell /usr/sbin/nologin quantinfer || true
sudo install -d -o quantinfer -g quantinfer -m 0750 \
  /opt/quantinfer/releases \
  /opt/quantinfer/models/qwen3 \
  /opt/quantinfer/models/source \
  /opt/quantinfer/jobs \
  /var/lib/quantinfer/logs
```

## 3. 获取并固定代码

```bash
sudo -u quantinfer git clone https://github.com/aiqubits/quantinfer.git /opt/quantinfer/src
cd /opt/quantinfer/src
git checkout f6a44e9fd265c5f06bf323cef02a2230a4ec5221
git rev-parse HEAD
```

构建前执行与 CI 相同的检查。

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo run -p quantinfer-xtask -- bench
cargo clippy --workspace --all-targets -- -D warnings
```

构建并安装到版本目录。`RELEASE_ID` 使用代码 commit 的短 SHA 或内部发布编号。

```bash
RELEASE_ID=f6a44e9
cargo build --release -p quantinfer-convert -p quantinfer-cli
sudo install -d -o quantinfer -g quantinfer -m 0755 "/opt/quantinfer/releases/$RELEASE_ID"
sudo install -m 0755 target/release/quantinfer-convert "/opt/quantinfer/releases/$RELEASE_ID/quantinfer-convert"
sudo install -m 0755 target/release/quantinfer-cli "/opt/quantinfer/releases/$RELEASE_ID/quantinfer-cli"
sudo ln -sfn "/opt/quantinfer/releases/$RELEASE_ID" /opt/quantinfer/current
test -x "/opt/quantinfer/current/quantinfer-convert"
test -x "/opt/quantinfer/current/quantinfer-cli"
readlink -f /opt/quantinfer/current/quantinfer-cli
```

## 4. 下载 Qwen3 0.6B 模型

当前转换器要求单个 SafeTensors 文件。`Qwen/Qwen3-0.6B` 提供单个 `model.safetensors`，适合作为当前流程的示例。

创建项目专用虚拟环境并安装 Hugging Face Hub CLI。

```bash
sudo -u quantinfer python3 -m venv /opt/quantinfer/.venv
sudo -u quantinfer /opt/quantinfer/.venv/bin/python -m pip install --upgrade pip huggingface_hub
```

公共模型无需登录。私有模型先执行以下命令。

```bash
sudo -u quantinfer /opt/quantinfer/.venv/bin/hf auth login
```

本示例固定 Hugging Face revision `c1899de289a04d12100db370d81485cdf75e47ca`，发布记录必须保留该值。

```bash
MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
sudo -u quantinfer mkdir -p /opt/quantinfer/models/source/qwen3-0.6b
sudo -u quantinfer /opt/quantinfer/.venv/bin/hf download Qwen/Qwen3-0.6B \
  config.json \
  tokenizer.json \
  tokenizer_config.json \
  merges.txt \
  vocab.json \
  model.safetensors \
  --revision "$MODEL_REVISION" \
  --local-dir /opt/quantinfer/models/source/qwen3-0.6b
```

记录模型文件校验和。

```bash
cd /opt/quantinfer/models/source/qwen3-0.6b
sha256sum config.json tokenizer.json tokenizer_config.json merges.txt vocab.json model.safetensors \
  | sudo tee input.sha256
```

Qwen3-1.7B、4B、8B 等模型通常使用多个 SafeTensors 分片。当前转换器不能直接读取 `model.safetensors.index.json` 或单个分片。

### Qwen3-4B-Instruct-2507 的适用性

[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 的 `config.json` 使用 `Qwen3ForCausalLM`、GQA、偶数 `head_dim`，并关闭 sliding-window 和 custom RoPE scaling，因此模型架构符合 QuantInfer 当前 Dense 校验。

它不适合作为当前 Runbook 的直接生产示例，原因如下。

- 官方仓库提供 `model.safetensors.index.json` 和多个权重分片，当前转换器不能直接读取。
- 4B 模型在 CPU Scalar 下的内存和延迟预算显著高于 0.6B，必须先用目标主机完成 RSS、Prefill 和 Decode 压测。
- 该模型的官方 chat template 由 `tokenizer_config.json` 提供，包含 Instruct 2507 的 system、tool call 和 assistant prompt 规则。当前 `Qwen3ChatTemplate` 是简化模板，并会在非 thinking 模式追加 `/no_think`，不能直接视为官方模板等价物。

因此，Runbook 的直接下载和转换示例使用 `Qwen3-0.6B`。如果目标是 4B-Instruct-2507，应先完成分片 SafeTensors 输入、官方 chat template 兼容和真实 tokenizer golden 校验，再把 4B 设为生产模型。

## 5. 转换为 Q8_0 GGUF

```bash
SOURCE=/opt/quantinfer/models/source/qwen3-0.6b
MODEL_DIR=/opt/quantinfer/models/qwen3/f6a44e9
sudo install -d -o quantinfer -g quantinfer -m 0750 "$MODEL_DIR"
/opt/quantinfer/current/quantinfer-convert \
  --config "$SOURCE/config.json" \
  --weights "$SOURCE/model.safetensors" \
  --tokenizer "$SOURCE/tokenizer.json" \
  --output "$MODEL_DIR/qwen3-q8_0.gguf" \
  --quantize q8_0
sha256sum "$MODEL_DIR/qwen3-q8_0.gguf" | sudo tee "$MODEL_DIR/qwen3-q8_0.gguf.sha256"
sudo ln -sfn "$MODEL_DIR" /opt/quantinfer/models/qwen3/current
```

转换器会校验 Qwen3 必需 Tensor、canonical 名称重复、head_dim、sliding-window、RoPE scaling、tokenizer vocab 和 SafeTensors shape/offset。

## 6. 检查 GGUF

```bash
/opt/quantinfer/current/quantinfer-cli \
  inspect-gguf \
  --input /opt/quantinfer/models/qwen3/current/qwen3-q8_0.gguf \
  | tee /var/lib/quantinfer/logs/inspect-qwen3.log
```

确认 Tensor inventory 包含关键 Tensor。

```bash
grep -E 'token_embd\.weight|output_norm\.weight|blk\.0\.attn_q\.weight|blk\.0\.ffn_down\.weight' \
  /var/lib/quantinfer/logs/inspect-qwen3.log
```

`inspect-gguf` 当前输出 metadata 数量和 Tensor inventory，不输出 metadata 键值。Qwen3 metadata 和 tokenizer metadata 会在下面的模型加载与 prompt smoke test 中被实际读取，缺失时命令会失败。

执行第 5 至第 8 节前，确认发布入口存在。缺少该文件时先完成第 3 节，不能直接运行推理命令。

```bash
test -x /opt/quantinfer/current/quantinfer-cli
```

## 7. 推理 smoke test

### 7.1 Token ID 测试

显式 token ID 测试模型加载、Prefill、Decode、KV Cache 和 EOS。

```bash
/opt/quantinfer/current/quantinfer-cli \
  generate \
  --input /opt/quantinfer/models/qwen3/current/qwen3-q8_0.gguf \
  --tokens 1,2,3 \
  --max-new-tokens 16 \
  --temperature 0 \
  --eos-token-id 151645 \
  | tee /var/lib/quantinfer/logs/smoke-token-ids.log
```

### 7.2 自然语言对话测试

`generate` 当前是非流式 CLI。`apps/quantinfer-cli/src/main.rs` 中的 `generate` 函数会完成模型加载、tokenizer 编码、Prefill 和 Decode，最后一次性打印 `prompt_tokens`、`generated_tokens` 和 `generated_text`。`--chat` 只调用 `Qwen3ChatTemplate` 生成 prompt，不会启用流式输出。stdout 重定向到文件时，终端不会显示结果，`cmp` 在两个文件相同时也不会输出任何文本。

使用 `tee` 同时保留日志和终端输出，并让 pipeline 传播 CLI 失败状态。

```bash
set -o pipefail
for run in 1 2; do
  if ! /opt/quantinfer/current/quantinfer-cli \
    generate \
    --input /opt/quantinfer/models/qwen3/current/qwen3-q8_0.gguf \
    --prompt "请用一句话说明 QuantInfer 的用途。" \
    --chat \
    --max-new-tokens 16 \
    --temperature 0 \
    2>&1 | tee "/var/lib/quantinfer/logs/smoke-prompt-${run}.log"; then
    exit 1
  fi
done
grep -E '^(prompt_tokens|generated_tokens|generated_text)=' \
  /var/lib/quantinfer/logs/smoke-prompt-1.log
cmp \
  <(grep '^generated_tokens=' /var/lib/quantinfer/logs/smoke-prompt-1.log) \
  <(grep '^generated_tokens=' /var/lib/quantinfer/logs/smoke-prompt-2.log) \
  && printf '%s\n' 'deterministic generated_tokens: PASS'
```

按以下顺序判断结果。

- 没有 `prompt_tokens=`，表示 tokenizer、Chat Template 或 GGUF tokenizer metadata 在 Prefill 前失败。查看日志中的 `quantinfer:` 错误。
- 有 `prompt_tokens=` 但没有 `generated_tokens=`，表示模型加载、Prefill 或 Decode 失败。检查 `quantinfer:` 错误和进程退出码。
- 有 `generated_tokens=` 但 `generated_text=` 为空，表示生成 token 全部被 `GgufBpeTokenizer::decode` 识别为 special token 并跳过。先用 token ID 检查生成结果，再检查 tokenizer metadata 与 EOS 配置。
- 进程仍在运行且日志为空，表示当前实现还没有流式输出，仍在执行 Prefill 或 Decode。先将 `--max-new-tokens` 降到 1 或 4，确认路径后再增加长度。

真实 tokenizer vectors 和真实 checkpoint logits 应在上线前加入回归门禁。

## 8. 资源基线

```bash
/usr/bin/time -v \
  /opt/quantinfer/current/quantinfer-cli \
  generate \
  --input /opt/quantinfer/models/qwen3/current/qwen3-q8_0.gguf \
  --prompt "health check" \
  --max-new-tokens 16 \
  --temperature 0 \
  > /var/lib/quantinfer/logs/resource-baseline.log
```

记录 CPU 型号、核心数、GGUF 大小、峰值 RSS、Prefill 时间和 Decode token 平均耗时。`quantinfer-xtask bench` 只验证 Q8_0 CPU 算子，不代表真实模型吞吐。

## 9. 上线边界

当前代码适合 CPU 单机、离线批处理和受控内网试运行。高并发在线服务需要常驻模型服务、多请求调度、真实 tokenizer golden 校验、真实 checkpoint 回归和性能压测。
