# QuantInfer Qwen3-0.6B CPU 单机试运行 Runbook

本文档适用于 Qwen3-0.6B Dense、CPU Scalar、单序列 Prefill/Decode 和 Q8_0 线性层量化。当前仓库提供 CLI 推理入口和可选的 HTTP 服务入口；本文只覆盖 CLI 单机流程。

本文 CLI 行为对照 commit `ee63330`。运行代码应固定为经过 CI 验证的 commit。发布时将本文示例中的提交替换为实际批准的 commit。

## 1. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| 模型 | Qwen3-0.6B Dense |
| 设备 | CPU Scalar |
| 输入权重 | 单个或索引描述的 SafeTensors 分片 |
| GGUF 浮点类型 | F32、F16、BF16，加载时转为 F32 |
| 量化 | Q8_0，量化线性投影层 |
| 生成 | 单序列 Prefill、Decode、KV Cache、EOS、temperature、seed |
| tokenizer | tokenizer.json 的 BPE vocab 和 merges |
| 服务 | CLI 单次生成 |

Qwen3 MoE、sliding-window 和 custom RoPE scaling 不在支持范围内。多请求调度由 `quantinfer-server` 提供，说明见 [`apps/README.md`](../../apps/README.md)；本 Runbook 只覆盖 CLI 单序列。CPU、WGPU、CUDA 和 macOS Metal 入口分别由对应 feature 提供。对照 commit：`ee63330`。

## 2. 主机准备

以下命令适用于 Debian/Ubuntu。模型、GGUF 和日志目录由 `quantinfer` 用户管理。

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ca-certificates coreutils python3-full python3-venv curl time
sudo useradd --system --home /opt/quantinfer --shell /usr/sbin/nologin quantinfer || true
sudo install -d -o quantinfer -g quantinfer -m 0750 \
  /opt/quantinfer/releases \
  /opt/quantinfer/models \
  /opt/quantinfer/jobs \
  /opt/quantinfer/.venv \
  /opt/quantinfer/.cache \
  /var/lib/quantinfer/logs
sudo install -d -o quantinfer -g quantinfer -m 0750 \
  /opt/quantinfer/models/qwen3 \
  /opt/quantinfer/models/source \
  /opt/quantinfer/.cache/pip
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
if [ -f "$HOME/.cargo/env" ]; then
  . "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustup toolchain install stable
rustup component add rustfmt clippy
```

## 3. 获取并固定代码

```bash
sudo git clone https://github.com/aiqubits/quantinfer.git /opt/quantinfer/src
cd /opt/quantinfer/src
git checkout <APPROVED_COMMIT>
git rev-parse HEAD
```

构建前执行与 CI 相同的检查。

```bash
cargo fmt --all -- --check
cargo check --workspace --locked
cargo test --workspace --locked
cargo run --locked -p quantinfer-xtask -- bench
cargo clippy --workspace --all-targets --locked -- -D warnings
```

构建并安装到版本目录。`RELEASE_ID` 使用代码 commit 的短 SHA 或内部发布编号。

```bash
RELEASE_ID=<APPROVED_COMMIT>
cargo build --release --locked -p quantinfer-convert -p quantinfer-cli
sudo install -d -o quantinfer -g quantinfer -m 0755 "/opt/quantinfer/releases/$RELEASE_ID"
sudo install -m 0755 target/release/quantinfer-convert "/opt/quantinfer/releases/$RELEASE_ID/quantinfer-convert"
sudo install -m 0755 target/release/quantinfer-cli "/opt/quantinfer/releases/$RELEASE_ID/quantinfer-cli"
sudo ln -sfn "/opt/quantinfer/releases/$RELEASE_ID" /opt/quantinfer/current
test -x "/opt/quantinfer/current/quantinfer-convert"
test -x "/opt/quantinfer/current/quantinfer-cli"
readlink -f /opt/quantinfer/current/quantinfer-cli
```

## 4. 下载 Qwen3 0.6B 模型

`Qwen/Qwen3-0.6B` 提供单个 `model.safetensors`，适合作为当前流程的示例。更大的模型可以使用 `model.safetensors.index.json`，转换器会按索引读取分片。

创建项目专用虚拟环境并安装 Hugging Face Hub CLI。

```bash
sudo install -d -o quantinfer -g quantinfer -m 0750 /opt/quantinfer/.venv
sudo install -d -o quantinfer -g quantinfer -m 0750 /opt/quantinfer/models /opt/quantinfer/models/source
sudo -H -u quantinfer python3 -m venv /opt/quantinfer/.venv
sudo -H -u quantinfer /opt/quantinfer/.venv/bin/python -m pip install --upgrade pip huggingface_hub
```

公共模型无需登录。私有模型先执行以下命令。

```bash
sudo -H -u quantinfer /opt/quantinfer/.venv/bin/hf auth login
```

本示例固定 Hugging Face revision `c1899de289a04d12100db370d81485cdf75e47ca`，发布记录必须保留该值。

```bash
MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
sudo install -d -o quantinfer -g quantinfer -m 0750 /opt/quantinfer/models/source/qwen3-0.6b
sudo -H -u quantinfer /opt/quantinfer/.venv/bin/hf download Qwen/Qwen3-0.6B \
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
sudo -H -u quantinfer sha256sum config.json tokenizer.json tokenizer_config.json merges.txt vocab.json model.safetensors \
  | sudo -H -u quantinfer tee input.sha256 >/dev/null
```

Qwen3-1.7B、4B、8B 等模型通常使用多个 SafeTensors 分片。转换器支持
`model.safetensors.index.json`，并按索引从对应分片读取张量。

### Qwen3-4B-Instruct-2507 的适用性

[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 的 `config.json` 使用 `Qwen3ForCausalLM`、GQA、偶数 `head_dim`，并关闭 sliding-window 和 custom RoPE scaling，因此模型架构符合 QuantInfer 当前 Dense 校验。

它不适合作为当前 Runbook 的直接示例，原因如下。

- 官方仓库提供 `model.safetensors.index.json` 和多个权重分片。使用索引文件作为
  `--weights` 参数，必要时通过 `--weights-checksums` 校验全部分片。
- 4B 模型在 CPU Scalar 下的内存和延迟预算显著高于 0.6B，必须先用目标主机完成 RSS、Prefill 和 Decode 压测。
- 该模型的官方 chat template 由 `tokenizer_config.json` 提供，包含 Instruct 2507 的 system、tool call 和 assistant prompt 规则。当前 CLI 仅构造单轮 Qwen3-0.6B prompt，`Qwen3ChatTemplate` 未覆盖 4B-Instruct-2507 的完整 system、tool call、assistant prompt 和多轮模板语义。

因此，Runbook 的直接下载和转换示例使用 `Qwen3-0.6B`。4B-Instruct-2507 仍需单独完成官方 chat template 兼容和真实 tokenizer golden 校验。

## 5. 转换为 Q8_0 GGUF

```bash
set -euo pipefail
SOURCE=/opt/quantinfer/models/source/qwen3-0.6b
MODEL_DIR=/opt/quantinfer/models/qwen3/<APPROVED_COMMIT>
sudo install -d -o quantinfer -g quantinfer -m 0750 "$MODEL_DIR"
sudo -H -u quantinfer /opt/quantinfer/current/quantinfer-convert \
  --config "$SOURCE/config.json" \
  --weights "$SOURCE/model.safetensors" \
  --tokenizer "$SOURCE/tokenizer.json" \
  --output "$MODEL_DIR/qwen3-q8_0.gguf" \
  --quantize q8_0
sudo -H -u quantinfer sha256sum "$MODEL_DIR/qwen3-q8_0.gguf" \
  | sudo -H -u quantinfer tee "$MODEL_DIR/qwen3-q8_0.gguf.sha256" >/dev/null
sudo -H -u quantinfer ln -sfn "$MODEL_DIR" /opt/quantinfer/models/qwen3/current
```

转换器会校验 SafeTensors 文件结构、Tensor 数据长度、Qwen3 配置约束、required Tensor 名称和 canonical 名称重复。完整 Tensor shape 语义会在 GGUF 模型加载时继续校验。

## 6. 检查 GGUF

```bash
set -o pipefail
sudo -H -u quantinfer /opt/quantinfer/current/quantinfer-cli \
  inspect-gguf \
  --input /opt/quantinfer/models/qwen3/current/qwen3-q8_0.gguf \
  | sudo -H -u quantinfer tee /var/lib/quantinfer/logs/inspect-qwen3.log
```

查看 Tensor inventory 中的关键 Tensor。

```bash
grep -E 'token_embd\.weight|output_norm\.weight|blk\.0\.attn_q\.weight|blk\.0\.ffn_down\.weight' \
  /var/lib/quantinfer/logs/inspect-qwen3.log
```

查看关键 metadata key。

```bash
grep -E 'metadata_key=(general\.architecture|qwen3\.context_length|qwen3\.embedding_length|qwen3\.vocab_size|qwen3\.feed_forward_length|qwen3\.block_count|qwen3\.attention\.head_count|qwen3\.attention\.head_count_kv|qwen3\.attention\.head_dim|qwen3\.rope\.freq_base|qwen3\.attention\.layer_norm_rms_epsilon|tokenizer\.ggml\.tokens|tokenizer\.ggml\.merges|tokenizer\.ggml\.eos_token_id)' \
  /var/lib/quantinfer/logs/inspect-qwen3.log
```

`inspect-gguf` 输出 metadata 数量、metadata key 和 Tensor inventory。模型加载阶段会读取必要 metadata 并校验其类型、必要字段和 Tensor shape，`inspect-gguf` 本身不校验 metadata 值。

## 7. 推理 smoke test

### 7.1 Token ID 测试

显式 token ID 测试模型加载、Prefill、Decode、KV Cache 和 EOS 参数处理。

```bash
set -o pipefail
sudo -H -u quantinfer /opt/quantinfer/current/quantinfer-cli \
  generate \
  --input /opt/quantinfer/models/qwen3/current/qwen3-q8_0.gguf \
  --tokens 1,2,3 \
  --max-new-tokens 16 \
  --max-sequence-length 4096 \
  --temperature 0 \
  --eos-token-id 151645 \
  | sudo -H -u quantinfer tee /var/lib/quantinfer/logs/smoke-token-ids.log
```

### 7.2 自然语言对话测试

`generate` 当前是非流式 CLI。`apps/quantinfer-cli/src/main.rs` 中的 `generate` 函数会完成模型加载、tokenizer 编码、Prefill 和 Decode，最后一次性打印 token、阶段耗时和 `generated_text`。`--chat` 生成 Qwen3-0.6B 单轮 prompt，不会启用流式输出。stdout 重定向到文件时，终端不会显示结果，`cmp` 在两个文件相同时也不会输出任何文本。该测试验证生成字段存在、`generated_text` 非空白且两次生成 token 一致，不验证回答事实正确性。

使用 `tee` 同时保留日志和终端输出，并让 pipeline 传播 CLI 失败状态。

```bash
set -euo pipefail
for run in 1 2; do
  if ! sudo -H -u quantinfer /opt/quantinfer/current/quantinfer-cli \
    generate \
    --input /opt/quantinfer/models/qwen3/current/qwen3-q8_0.gguf \
    --prompt "请用一句话说明 QuantInfer 的用途。" \
    --chat \
    --max-new-tokens 16 \
    --max-sequence-length 4096 \
    --temperature 0 \
    2>&1 | sudo -H -u quantinfer tee "/var/lib/quantinfer/logs/smoke-prompt-${run}.log"; then
    exit 1
  fi
done
grep -E '^(prompt_tokens|generated_tokens|generated_text)=' \
  /var/lib/quantinfer/logs/smoke-prompt-1.log
python3 - <<'PY'
from pathlib import Path

for run in (1, 2):
    log = Path(f"/var/lib/quantinfer/logs/smoke-prompt-{run}.log").read_text()
    for marker in ("prompt_tokens=", "generated_tokens=", "generated_text="):
        if marker not in log:
            raise SystemExit(f"missing {marker} in smoke-prompt-{run}.log")
    generated_text = log.split("generated_text=", 1)[1]
    if not generated_text.strip():
        raise SystemExit(f"generated_text is empty or whitespace in smoke-prompt-{run}.log")
PY
cmp \
  <(grep '^generated_tokens=' /var/lib/quantinfer/logs/smoke-prompt-1.log) \
  <(grep '^generated_tokens=' /var/lib/quantinfer/logs/smoke-prompt-2.log) \
  && printf '%s\n' 'deterministic generated_tokens: PASS'
```

按以下顺序判断结果。

- 没有 `prompt_tokens=`，表示 tokenizer、Chat Template 或 GGUF tokenizer metadata 在 Prefill 前失败。查看日志中的 `quantinfer:` 错误。
- 有 `prompt_tokens=` 但没有 `generated_tokens=`，表示模型加载、Prefill 或 Decode 失败。检查 `quantinfer:` 错误和进程退出码。
- 有 `generated_tokens=` 但 `generated_text=` 为空或只含空白，表示生成结果可能只包含 special token、换行或其它空白字符。先检查 `generated_tokens`，再检查 tokenizer metadata、Chat Template 和 EOS 配置。
- 进程仍在运行且日志为空，表示当前实现还没有流式输出，仍在执行 Prefill 或 Decode。先将 `--max-new-tokens` 降到 1 或 4，确认路径后再增加长度。

真实 tokenizer vectors 和真实 checkpoint logits 应纳入后续版本的回归门禁。

## 8. 资源基线

```bash
set -o pipefail
sudo -H -u quantinfer /usr/bin/time -v \
  /opt/quantinfer/current/quantinfer-cli \
  generate \
  --input /opt/quantinfer/models/qwen3/current/qwen3-q8_0.gguf \
  --prompt "health check" \
  --max-new-tokens 16 \
  --max-sequence-length 4096 \
  --temperature 0 \
  2>&1 | sudo -H -u quantinfer tee /var/lib/quantinfer/logs/resource-baseline.log >/dev/null
```

记录 CPU 型号、核心数、GGUF 大小、峰值 RSS、`prefill_ms`、`decode_ms`、`decode_tokens` 和 `decode_tokens_per_sec`。`quantinfer-xtask bench` 只验证 Q8_0 CPU 算子，不代表真实模型吞吐。

`--max-sequence-length` 限制 KV Cache 的最大序列长度。Qwen3-0.6B 的模型配置是 40960，keys 和 values 使用 F32 时理论容量约为 8.75 GiB。目标主机应先使用 4096 或更小值完成内存基线，再按业务上下文需求逐步增加。

## 9. 适用边界

当前代码适合 Qwen3-0.6B CPU 单机、离线单序列任务和受控内网试运行。`quantinfer-server` 已提供排队、连续批处理和流式响应，但不在本文流程内，也不构成完整的在线生产平台。
