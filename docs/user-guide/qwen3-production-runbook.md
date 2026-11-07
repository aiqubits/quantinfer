# QuantInfer Qwen3 量化与推理生产 Runbook

本文档对应 QuantInfer `main` 的生产候选提交。部署前固定一个经过 CI 验证的 commit，例如 `724c53f006d5f235867ebdd8c4e5e63f33ddbdf6`。

当前实现适合单机 CPU、单序列、离线批处理和受控内网试运行。`quantinfer-server` 仍是 scaffold，仓库没有常驻 HTTP 服务、请求队列、多会话调度或模型常驻 API。在线生产服务需要在本 runbook 的上线门禁通过后，再接入自己的进程管理和服务层。

## 1. 支持矩阵

| 项目 | 当前支持 | 生产含义 |
| --- | --- | --- |
| 模型 | Qwen3 Dense | Qwen3 MoE、滑动窗口和自定义 RoPE scaling 会被拒绝 |
| 设备 | CPU Scalar | 需要 CPU 运行，CUDA、Metal、WGPU、Ascend 后端仍是 scaffold |
| 输入权重 | 单个 SafeTensors 文件 | 分片 SafeTensors 需要在 QuantInfer 外部合并 |
| GGUF 浮点类型 | F32、F16、BF16 | 加载时转成 F32 |
| 量化类型 | Q8_0 | 只量化线性投影层 |
| tokenizer | tokenizer.json 中的 BPE vocab 和 merges | 需要执行真实 Qwen3 tokenizer golden 校验 |
| 生成 | Prefill、Decode、KV Cache、EOS、temperature、seed | 单序列生成 |
| 服务 | CLI 一次执行一次生成 | 暂无 QuantInfer 原生 HTTP 服务 |

## 2. 主机准备

建议使用 Linux x86_64 主机。构建阶段需要 Rust stable、`rustfmt` 和 `clippy`。仓库中的 `rust-toolchain.toml` 会声明这些组件。

```bash
sudo apt-get update
sudo apt-get install -y build-essential git ca-certificates jq coreutils
rustup toolchain install stable
rustup component add rustfmt clippy
```

为模型、转换产物和日志分开准备目录。模型目录设为只读，转换目录和日志目录由部署用户拥有。

```bash
sudo useradd --system --home /opt/quantinfer --shell /usr/sbin/nologin quantinfer || true
sudo install -d -o quantinfer -g quantinfer -m 0750 \
  /opt/quantinfer/releases \
  /opt/quantinfer/models/qwen3 \
  /var/lib/quantinfer/logs
```

部署账号不需要 root 权限。模型目录只允许发布流程写入。

## 3. 获取并固定代码

```bash
git clone https://github.com/aiqubits/quantinfer.git /opt/quantinfer/src
cd /opt/quantinfer/src
git checkout 724c53f006d5f235867ebdd8c4e5e63f33ddbdf6
git status --short
```

构建前执行仓库 CI 的本地等价检查。生产发布记录必须保存 commit、Rust toolchain、主机 CPU 型号和构建日志。

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo run -p quantinfer-xtask -- bench
cargo clippy --workspace --all-targets -- -D warnings
```

生成 release 二进制。

```bash
cargo build --release -p quantinfer-convert -p quantinfer-cli
install -m 0755 target/release/quantinfer-convert /opt/quantinfer/releases/quantinfer-convert
install -m 0755 target/release/quantinfer-cli /opt/quantinfer/releases/quantinfer-cli
```

## 4. 准备 Qwen3 模型文件

转换器需要以下输入。

```text
config.json              Qwen3 Dense 配置
model.safetensors        合并后的单个 SafeTensors 权重文件
tokenizer.json           BPE model.vocab 和 model.merges
```

`config.json` 至少需要包含以下字段。实际 Qwen3 配置可以包含更多字段。

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "vocab_size": 151936,
  "hidden_size": 1024,
  "intermediate_size": 3072,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "rms_norm_eps": 0.000001,
  "rope_theta": 1000000.0,
  "max_position_embeddings": 40960,
  "tie_word_embeddings": true,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "use_sliding_window": false
}
```

发布前核对权重命名包含以下 HF 命名集合。转换器会映射成 canonical GGUF 名称。

```text
model.embed_tokens.weight
model.norm.weight
model.layers.*.input_layernorm.weight
model.layers.*.self_attn.q_proj.weight
model.layers.*.self_attn.q_norm.weight
model.layers.*.self_attn.k_proj.weight
model.layers.*.self_attn.k_norm.weight
model.layers.*.self_attn.v_proj.weight
model.layers.*.self_attn.o_proj.weight
model.layers.*.post_attention_layernorm.weight
model.layers.*.mlp.gate_proj.weight
model.layers.*.mlp.up_proj.weight
model.layers.*.mlp.down_proj.weight
```

如果模型是分片 SafeTensors，先在模型准备阶段合并为一个文件，再进入本流程。不要把分片中的单个文件直接交给转换器。

## 5. 执行 Q8_0 转换

先计算输入文件校验和，并把校验和写入发布记录。

```bash
sha256sum config.json model.safetensors tokenizer.json \
  | tee /opt/quantinfer/models/qwen3/input.sha256
```

执行转换。`--tokenizer` 会把 vocab、merges、BOS 和 EOS 元数据写入 GGUF。`--quantize q8_0` 只量化线性投影层，embedding、RMSNorm 和 lm head 保持浮点。

```bash
/opt/quantinfer/releases/quantinfer-convert \
  --config config.json \
  --weights model.safetensors \
  --tokenizer tokenizer.json \
  --output /opt/quantinfer/models/qwen3/qwen3-q8_0.gguf \
  --quantize q8_0
```

转换成功后记录输出校验和。

```bash
sha256sum /opt/quantinfer/models/qwen3/qwen3-q8_0.gguf \
  | tee /opt/quantinfer/models/qwen3/qwen3-q8_0.gguf.sha256
```

转换器会拒绝以下情况。

- Qwen3 必需 Tensor 缺失。
- 多个输入 Tensor 映射到同一个 canonical GGUF 名称。
- `head_dim` 为奇数。
- sliding-window 或自定义 RoPE scaling 被启用。
- tokenizer vocab 超过模型 vocab。
- SafeTensors shape、offset 或实际字节数不一致。

## 6. 检查 GGUF 产物

先检查 metadata 和 Tensor inventory。

```bash
/opt/quantinfer/releases/quantinfer-cli \
  inspect-gguf \
  --input /opt/quantinfer/models/qwen3/qwen3-q8_0.gguf \
  | tee /var/lib/quantinfer/logs/inspect-qwen3.log
```

检查结果至少应满足以下条件。

```bash
rg 'token_embd\.weight|output_norm\.weight|blk\.0\.attn_q\.weight|blk\.0\.ffn_down\.weight' \
  /var/lib/quantinfer/logs/inspect-qwen3.log
```

还要确认以下 metadata 存在于产物中。`inspect-gguf` 当前打印 metadata 数量和 Tensor 列表，详细 metadata 可通过 QuantInfer format API 或后续专用 inspect 输出核对。

```text
general.architecture = qwen3
qwen3.context_length
qwen3.embedding_length
qwen3.vocab_size
qwen3.feed_forward_length
qwen3.block_count
qwen3.attention.head_count
qwen3.attention.head_count_kv
qwen3.attention.head_dim
qwen3.rope.freq_base
qwen3.attention.layer_norm_rms_epsilon
tokenizer.ggml.tokens
tokenizer.ggml.merges
tokenizer.ggml.eos_token_id
```

## 7. 运行 CPU 生成 smoke test

### 7.1 显式 token ID 测试

这个模式用于排除 tokenizer 影响，确认 GGUF 加载、Prefill、Decode、KV Cache 和 EOS 停止。

```bash
/opt/quantinfer/releases/quantinfer-cli \
  generate \
  --input /opt/quantinfer/models/qwen3/qwen3-q8_0.gguf \
  --tokens 1,2,3 \
  --max-new-tokens 16 \
  --temperature 0 \
  --eos-token-id 151645 \
  | tee /var/lib/quantinfer/logs/smoke-token-ids.log
```

### 7.2 自然语言 prompt 测试

这个模式会从 GGUF metadata 构造 byte-level BPE tokenizer，并可应用 Qwen3 chat template。

```bash
for run in 1 2; do
  /opt/quantinfer/releases/quantinfer-cli \
    generate \
    --input /opt/quantinfer/models/qwen3/qwen3-q8_0.gguf \
    --prompt "请用一句话说明 QuantInfer 的用途。" \
    --chat \
    --max-new-tokens 64 \
    --temperature 0 \
    --seed 42 \
    > "/var/lib/quantinfer/logs/smoke-prompt-${run}.log"
done
cmp \
  <(grep '^generated_tokens=' /var/lib/quantinfer/logs/smoke-prompt-1.log) \
  <(grep '^generated_tokens=' /var/lib/quantinfer/logs/smoke-prompt-2.log)
```

生产环境应使用真实 Qwen3 tokenizer vectors 对比 HuggingFace 结果，再把自然语言 prompt 入口纳入上线门禁。当前仓库还没有提交真实 checkpoint 和官方 golden vectors。

## 8. 资源和性能基线

QuantInfer 当前使用 CPU Scalar backend，模型文件整体读入内存，线性层在运行时执行 Q8_0 matmul。上线前使用目标机器测量，而不是用开发机估算。

```bash
/usr/bin/time -v \
  /opt/quantinfer/releases/quantinfer-cli \
  generate \
  --input /opt/quantinfer/models/qwen3/qwen3-q8_0.gguf \
  --prompt "health check" \
  --max-new-tokens 16 \
  --temperature 0 \
  > /var/lib/quantinfer/logs/resource-baseline.log
```

同时运行内置 smoke benchmark。

```bash
cd /opt/quantinfer/src
cargo run --release -p quantinfer-xtask -- bench
```

记录以下数据。

- CPU 型号和核心数。
- GGUF 文件大小。
- 进程峰值 RSS。
- Prefill 时间。
- 每个 Decode token 的平均耗时。
- 失败率和 EOS 命中率。

当前 benchmark 是 Q8_0 CPU 算子 smoke check，不能替代真实模型吞吐和延迟压测。

## 9. 受控单机运行方式

当前 CLI 是一次性进程。每次执行都会重新读取 GGUF 和构造模型，因此适合离线任务、批处理、canary 和受控内网验证。它不适合作为高 QPS 在线服务的常驻请求进程。

可以用 systemd 管理一个受控批处理任务模板。

```ini
# /etc/systemd/system/quantinfer-qwen3-batch@.service
[Unit]
Description=QuantInfer Qwen3 batch job %i
After=local-fs.target

[Service]
Type=oneshot
User=quantinfer
Group=quantinfer
WorkingDirectory=/opt/quantinfer
ExecStart=/opt/quantinfer/releases/quantinfer-cli generate --input /opt/quantinfer/models/qwen3/qwen3-q8_0.gguf --prompt %i --chat --max-new-tokens 64 --temperature 0 --seed 42
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadOnlyPaths=/opt/quantinfer/models /opt/quantinfer/releases
ReadWritePaths=/var/lib/quantinfer/logs
MemoryMax=32G
TasksMax=128
TimeoutStartSec=30m
StandardOutput=journal
StandardError=journal
```

启动一个批处理任务。

```bash
sudo systemctl start 'quantinfer-qwen3-batch@请解释Q8_0.service'
journalctl -u 'quantinfer-qwen3-batch@请解释Q8_0.service' -f
```

prompt 中包含 systemd instance 不允许的特殊字符时，改用受控脚本读取 prompt 文件，并通过固定参数调用 CLI。不要把未经过滤的外部请求直接拼接到 shell 命令中。

## 10. 容器化建议

生产镜像使用固定 commit 构建，模型通过只读 volume 注入，运行用户不使用 root。容器入口执行一次生成任务或内部批处理 wrapper。

```dockerfile
FROM debian:bookworm-slim
RUN useradd --system --uid 10001 quantinfer
COPY target/release/quantinfer-cli /usr/local/bin/quantinfer-cli
RUN chmod 0555 /usr/local/bin/quantinfer-cli
USER quantinfer
ENTRYPOINT ["/usr/local/bin/quantinfer-cli"]
```

```bash
docker run --rm \
  --read-only \
  --cpus=8 \
  --memory=32g \
  -v /opt/quantinfer/models/qwen3:/models:ro \
  quantinfer:qwen3-q8_0 \
  generate --input /models/qwen3-q8_0.gguf --prompt "health check" --max-new-tokens 16 --temperature 0
```

内存限制必须以目标模型的 `/usr/bin/time -v` 结果为依据。容器启动后应先执行 health prompt，再接收业务任务。

## 11. 上线门禁

以下条件全部通过后再切换生产流量。

- 代码 commit、Rust toolchain、二进制 sha256 已记录。
- GitHub Actions 的 fmt、check、test、benchmark 和 Clippy 全部成功。
- 输入 config、SafeTensors、tokenizer.json 校验和已记录。
- GGUF 转换成功，required Tensor 检查通过，Tensor 名称无重复。
- `inspect-gguf` 输出包含 Qwen3 metadata 和 tokenizer metadata。
- 显式 token ID smoke test 成功。
- 自然语言 prompt smoke test 成功。
- 相同 prompt 在 temperature 0 下重复输出一致。
- 目标机器 RSS、Prefill、Decode 延迟满足业务预算。
- 真实 Qwen3 tokenizer 与 HuggingFace 的 golden vectors 通过。
- 真实或等价 Qwen3 checkpoint 的 logits、EOS 和短 prompt 输出通过回归校验。
- 生产运行账号无写模型目录权限。
- 失败时可以切回上一版二进制和 GGUF。

## 12. 发布与回滚

按版本目录保存二进制和模型，用符号链接切换当前版本。

```text
/opt/quantinfer/releases/2026-11-07-724c53f/
/opt/quantinfer/models/qwen3/2026-11-07-q8_0/
/opt/quantinfer/current -> /opt/quantinfer/releases/2026-11-07-724c53f
/opt/quantinfer/models/qwen3/current -> /opt/quantinfer/models/qwen3/2026-11-07-q8_0
```

发布时先切换 canary 主机，执行第 7 节 smoke test 和第 11 节门禁，再切换 `current`。回滚时恢复上一版符号链接，保留失败版本的日志、校验和及资源基线。

## 13. 当前上线结论

QuantInfer 当前可以作为 Qwen3 Dense CPU 量化推理的离线和受控试运行组件上线。互联网可访问的高并发在线服务还需要常驻 server、模型复用、多请求调度、真实模型 golden 验证、tokenizer 完整兼容和性能压测结果。
