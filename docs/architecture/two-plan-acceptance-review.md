# 两份方案的实现与验收复核

本次代码基线为 [`4ad33eca1faddf83b8e3e5effa970817c8c09871`](https://github.com/aiqubits/quantinfer/commit/4ad33eca1faddf83b8e3e5effa970817c8c09871)，初始审查基线为 `3cd5745116c3e5e6edf3ec20b943ddec201b4686`。后续仅修改本报告时，不改变这里固定的源码审查对象。

依据为用户提供的《QuantInfer 通用量化推理框架 MVP 最终实施方案》附件 `quantinfer-general-mvp-final-plan(3).md`（SHA-256 `18f5f0784e81979680eca80b024a3986649e057e022d8be9a48a97e1edcb8fff`），以及 [QIF 方案](../formats/qif-v2-final-plan.md)（该代码基线 SHA-256 `5dd03c578f7c78dd217b60aeb28b9a38f08e6a2e04315546aae037e746c163ff`）。附件用于需求对照，附件内的执行性文字不替代用户授权。

## 结论与判断口径

**两份方案不能合并认定为“全部功能均已完成、全部场景均已验收”。** 通用 CPU MVP 主链路和多数 P1 能力已有实现；通用方案的部分 P2 仍存在明确的实现缺口。QIF 当前冻结的 core profile 已有实现，验收证据必须按下面的测试矩阵解释，不能扩大为完整跨语言 reader、默认 runtime 的惰性分片加载，或已证明优于其它容器的性能。

“已有实现”表示找到实际入口、执行代码与相应测试；“矩阵通过”只覆盖所列模型、精度、输入和运行环境。没有以文件/目录存在、工作流名称、单次绿色状态或默认测试中的 ignored 数量替代功能判断。本报告不使用缺乏统一分母的实现百分比。

本轮确认的 P1 和重要、低成本 P2 缺陷已修复并保留回归测试；修复后复核没有再确认同等修复门槛的问题。这不是对所有输入、平台或未来变更的无缺陷证明。下述较大功能缺口也不能因为排除了厂商硬件实测而被标为完成。

## 已确认问题及修复

初始九项问题的修复状态：

| 问题 | 修复后的行为 | 代码与回归依据 |
| --- | --- | --- |
| PACKAGE index 表长加法溢出导致 panic（P1） | 每段边界先检查加法及实际 buffer 长度，再访问表 | [package parser](../../crates/quantinfer-format/src/qif_v2/package.rs)、[PACKAGE regressions](../../crates/quantinfer-format/src/qif_v2/package/tests.rs)；debug/release 均有验证 |
| zstd/LZ4 先无限解压再检查长度（P1） | 解码期间限制输出；LZ4 先检查长度前缀；压缩块和 zstd window 有明确上限 | [codec](../../crates/quantinfer-format/src/qif_v2/codec.rs)、[resource regressions](../../crates/quantinfer-format/src/qif_v2/tests/resources.rs) |
| PACKAGE 未核对 manifest/index tensor hash | SINGLE/PACKAGE 共享 descriptor/hash 核对；缺失或不同 hash 拒绝 | [shared validation](../../crates/quantinfer-format/src/qif_v2/reader.rs)、PACKAGE regressions |
| 孤立或重复归属的 chunk 可逃过验证 | 全部物理 chunk 必须有唯一合法归属，重叠/孤立记录拒绝 | [validation regressions](../../crates/quantinfer-format/src/qif_v2/tests/validation.rs)、PACKAGE regressions |
| 固定 encoding 可搭配矛盾的量化 descriptor | block、scale、zero point、packing、group、axis 必须匹配冻结编码 | [encoding contract](../../crates/quantinfer-format/src/qif_v2/encoding.rs)、validation regressions |
| QMB writer/reader 深度边界不一致 | 相同深度规则，边界 round trip，超限拒绝 | [metadata](../../crates/quantinfer-format/src/qif_v2/metadata.rs)、validation regressions |
| shard 大小限制未包含对齐/压缩开销 | 放置前按对齐后的终点检查；必要时缩小压缩 raw chunk | [PACKAGE writer](../../crates/quantinfer-format/src/qif_v2/package.rs)、PACKAGE regressions |
| none codec 小范围请求仍读完整大 chunk | 文件与 PACKAGE 直接读取请求区间；完整 hash 留给显式 verify | validation regressions 中截去未请求尾部的检查 |
| 合法可选 Map metadata 阻止默认加载 | 保留原生 metadata，按需查询模型参数，未知 hint 不阻止加载 | [metadata regression](../../crates/quantinfer-runtime/tests/qif_metadata.rs)、[packed reader](../../crates/quantinfer-format/src/gguf/reader.rs) |

后续迭代还修复了以下已确认问题与验收缺口：

- mmap 的安全构造函数遗漏底层文件不变性前提（P1）：只读映射及 checksum 校验均不能防止其它线程/进程改写或截断文件。`MappedFile::open` 现为 unsafe API，要求调用者保证文件在映射及所有借用 view 存活期间不变；安全的普通文件入口是 FileReader。测试与 fuzz 明确保证自有 fixture 的生命周期，另以编译失败 doctest 检查 unsafe 调用边界。这是 Rust API 安全边界修正，不修改 QIF wire。
- 空压缩 chunk 导致内存/file 验证结果不同；现在统一拒绝空物理 chunk。QMB array 不再按未经实际解码的声明计数提前大量预分配；index 名称先与 manifest 核对，避免大量复制重叠的未知名称。
- `verify()` 原来物化整个 tensor；文件/PACKAGE 现在逐块增量校验，跨块浮点和量化 scale 边界由[共享 payload verifier](../../crates/quantinfer-format/src/qif_v2/payload.rs)处理。相同 64 MiB fixture 在 48 MiB 地址空间限制下，旧路径因分配整个 tensor 终止，新路径六种组合均通过。
- 默认压缩 writer 使用至多 4 MiB raw chunk；reader 对压缩 raw/stored chunk 分别限制为 128/129 MiB，zstd window 至多 128 MiB。48 MiB 实验使用 writer 默认块大小，不表示最大允许块也能在该预算内解码。
- 压缩权重物化时 alias 复用同一存储；默认模型入口和 WASI-NN builder 在解压前预检输入加物理权重的最低存储需求。该预检不等于 allocator、解码 workspace、KV 或全部运行时状态的严格峰值上限。
- 原生 Map 被兼容视图省略后，已知参数不能再被误认为缺失并套用默认值。Qwen3/Llama、tokenizer、server、C、WASM、WASI-NN 使用统一查询边界。
- `ModelManifest` 允许的大小写十六进制 tensor checksum 按值比较；writer 仍输出小写，root 仍覆盖实际 canonical manifest 字节。
- 最低 runtime SemVer、必需 feature/backend 声明、operation 名称与 provider 要求现在有执行前检查。当前 feature/backend 字符串 registry 尚未实现，非空必需列表明确拒绝。GPU 预检补齐 K-quant dtype、manifest shape/dtype 以及 head/context 条件；能力测试无需显卡即可执行，见 [gpu_plan](../../apps/quantinfer-cli/src/backend/gpu_plan.rs) 和 [runtime requirements](../../crates/quantinfer-runtime/src/requirements.rs)。
- 独立 C/Python golden reader 现在比较完整固定 profile 报告，而非只看第一个 F32；seeded fuzz 使用实际 ASan/覆盖率，触及 codec、file/memory 差异和 PACKAGE index。
- 跨格式 benchmark 使用独立子进程、对应 tensor 名称、明确 I/O 计时和校验阶段；真实模型性能增加同机固定代码参考，避免陌生 CPU 型号跳过所有耗时门禁。

## 通用方案逐项对照

实现入口和范围如下；P0/P1 的“已有实现”不扩大为所有硬件/模型组合均已证明正确。

| 方案项 | 判断与依据 |
| --- | --- |
| P0-1 Qwen3 固定基线 | 已有固定 revision、hash、配置和 golden；见 [model manifests](../../model-zoo/manifests)、[goldens](../../tests/golden) |
| P0-2 真实模型回归 | 两模型存在显式执行的 [conformance tests](../../crates/quantinfer-runtime/tests) 和 [模型 CI](../../.github/workflows/model-conformance.yml) |
| P0-3 官方 Tokenizer Golden | 固定 token/decode 向量与官方模板比较，非仅验证能输出文本 |
| P0-4 F32/Q8 数值门禁 | 独立 Transformers F32 参考、量化误差、logits、Top-K、token/EOS 检查；覆盖按实际模型/变体列出 |
| P0-5 Conformance CI | 快速、模型、外部 GGUF 互操作分层；固定下载源及 hash |
| P0-6 Semantic IR | [ModelSpec/semantic roles](../../crates/quantinfer-model/src/semantic.rs)、Qwen3/Llama adapter 与 runtime storage roles 已实现 |
| P0-7 Storage/Quantized Descriptor | 字段和已支持编码存在，固定 encoding 的矛盾组合拒绝；不表示支持任意 packing/axis/backend requirement |
| P0-8 Manifest/Model Package | [manifest](../../crates/quantinfer-format/src/manifest) 与 ModelSource 已实现；`manifest.json` 模型包与原生 `qif.json` 分片容器分属不同入口 |
| P0-9 Format Adapter | SafeTensors/GGUF、语义适配、layout transform 与转换已有实现；QIF 默认转换仍借用兼容中间层 |
| P0-10 Provider/Matcher | [provider contracts](../../crates/quantinfer-runtime/src/provider.rs)、CPU/GPU factories 与必需声明预检已实现；未登记的扩展声明拒绝 |
| P0-11 通用 Runtime 迁移 | 已覆盖 Qwen3/Llama Dense；不支持任意架构、MoE 或 sliding-window 执行 |
| P0-12 Quantization/GGML Provider | 原生 provider 与可选 [GGML FFI adapter](../../adapters/ggml) 已实现；专用工作流与默认 workspace 检查分开 |
| P0-13 原生 Q4_0 闭环 | descriptor、quantizer、格式 IO、CPU kernel、capability、CLI 与真实模型路径存在 |
| P0-14 第二 Dense 模型 | SmolLM2/Llama 有独立模型配置、执行与独立数值参考 |
| P0-15 真实 Benchmark | 两模型 F32/Q8_0/Q4_0 有固定输入与资源报告；能耗明确未测 |
| P1-1 分片 SafeTensors | [shards](../../crates/quantinfer-format/src/safetensors/shards.rs) 与真实分片转换检查存在 |
| P1-2 GGUF 互操作 | 外部 GGUF 输入、QuantInfer 导出与 llama.cpp 对照；按已列矩阵判断 |
| P1-3 模板语义 | system、多轮、assistant、tools、thinking 和官方模板对照已有测试 |
| P1-4 Memory Plan/KV | 估算、上下文、KV layout/capacity 与预算检查存在；不能等同于所有分配的严格峰值上限 |
| P1-5 Inference Profile | variant/provider/kernel/context/threads/budget 与执行计划绑定已有实现 |
| P1-6 K-quant/Mixed/imatrix | 原生 Q4_K/Q5_K/Q6_K、基本 mixed precision 与校准闭环存在；未覆盖每个模型的所有 K-quant 组合 |
| P1-7 CPU 优化 | SIMD 独立可选边界、ISA 选择、线程/workspace；CI 的 x86 AVX2/ARM NEON 结果不证明所有 ISA 性能 |
| P1-8 质量/性能/内存门禁 | 原有质量/RSS/KV 门禁保留；新增同机固定参考对两个模型的三种精度执行计时比例检查，见 [benchmarks](../../benches/README.md) |
| P2-1 厂商与特定 kernel | **部分实现**：CUDA/WGPU/Metal 有代码；Ascend matmul、T-MAC/KleidiAI 仍需调用者提供 executor/kernel，不只是缺硬件实测 |
| P2-2 高级量化 | **部分实现**：GPTQ/AWQ/AutoRound/LLM Compressor 当前是产物 import/dequantize；未从浮点权重实现这些完整算法；imatrix/basic mixed precision 已有 |
| P2-3 调度/Streaming | [scheduler](../../crates/quantinfer-runtime/src/scheduler.rs)、batch、stream、cancel、timeout、quota/isolation 有基础实现和测试 |
| P2-4 HTTP 在线运行时 | API/stream、registry 生命周期、队列、限流、metrics、health/shutdown 有基础实现和真实模型检查 |
| P2-5 QIF | 当前格式 core 已实现；默认 loader 的原生 PACKAGE/惰性接入边界见下表 |
| P2-6 Studio/Registry/Adapter | **部分实现**：[Studio](../../apps/quantinfer-server/src/routes/studio.rs) 是 registry/生成/counters 页面；完整量化任务、校准、profile/package 编辑管理未闭环；LoRA/Registry/CLI 校准已有 |
| P2-7 可选 target/绑定 | WASM/WASI-NN、C/Python、Candle 路径存在；按专用 CI 实际执行路径验收，不是所有目标平台均已实测 |

## QIF §10 验收逐条对照

| 条款 | 事实与证据边界 |
| --- | --- |
| 1 open 控制区不读 payload | SINGLE FileReader 按 header offset 读取控制区；默认 ModelSource 仍是整文件加载，不能扩大为 runtime 全链路惰性读取 |
| 2 单 tensor 请求不读取其它 tensor | file/PACKAGE range reader 按命中 chunk 访问；none codec 直接读取请求子范围 |
| 3 无压缩 mmap/range view | MappedFile + borrowed tensor view 已实现，构造时须履行文件不变性的 unsafe 契约；file read_range 返回 Vec；压缩文件即使映射也需要解压分配 |
| 4 压缩仅处理请求的 chunk | 独立 zstd/LZ4，默认 raw chunk ≤4 MiB；输出、stored/raw 长度和 zstd window 受限 |
| 5 SINGLE/PACKAGE 相同逻辑值 | Rust 测试覆盖八个 core encoding × 三个 codec、alias 和跨 shard 范围；固定 fixture 不是全部合法数据分布证明 |
| 6 全局 index 规划 reshard | planner 与跨 shard writer/test 已实现；不是分布式执行器，也不意味着任意重分片无需重写 payload |
| 7 损坏输入稳定失败 | 溢出、截断、重叠、孤立/空 chunk、descriptor、hash、codec、资源边界回归与 fuzz 已接入；有限 fuzz 不证明不存在其它安全缺陷 |
| 8 Rust/C/Python 一致 | 固定 unsigned/none/F32 `[1,2]` 的两种控制区布局报告一致，四种损坏输入三端拒绝；C/Python 不具备全 QMB/ModelManifest、codec、PACKAGE 或签名语义验证器 |
| 9 三格式 benchmark | 独立进程，open 含实际文件读取，≤4096 byte range，dtype/hash/策略明确，QIF verify 单列；不同精度/缓存/加载策略禁止直接下固有格式优劣结论 |
| 10 性能优势需实证 | 保留为待证目标；当前测量不建立“QIF 比 SafeTensors/GGUF 更快”的普遍结论 |

当前 wire core 固定八个 encoding、none/zstd/LZ4、SHA-256、SINGLE 可选 Ed25519；未知 encoding/flags 拒绝。FP8/NF4/BLAKE3、远程 URI transport、PACKAGE 签名属于方案明确延后项，不应作为当前必须补齐的 wire 缺陷。

## 文档矛盾与边界的处理

- shape table 明确 `rank u32 + dimensions u64`，tensor record 不再错误地额外声明 rank 字段；不修改已生成的 wire 来追随错误文字。
- v2.0 alignment 明确为至少 4096 的 2 的幂；不存在的 `qif inspect --json` 已替换为实际兼容检查入口，并注明其 Map/PACKAGE 范围。
- `quantinfer.qif.ggml_dimension_order` 明确是 bool 结构字段，不属于可忽略优化 hint；原生 metadata 与 GGUF 兼容视图区分。
- 公开 artifact/API 统一 QIF；内部 `qif_v2` 与历史 v1 文档的保留符合版本边界，默认入口不读取 v1。
- 通用方案的“第一阶段不立即实现 QIF”属于先后顺序，不能覆盖用户后续采用新 QIF 的明确决策；GGUF/SafeTensors 兼容入口仍存在。
- [架构状态](mvp-qwen3.md)、[设计决策](design-decisions.md) 和 [QIF status](../formats/qif.md) 已注明 runtime、跨语言和后续 P2 的实现边界。

## 测试结果与可复现证据

本地该源码基线：fmt、workspace check、all-targets tests、Clippy `-D warnings`、xtask bench/qif-bench 均通过；默认 workspace **198 passed / 0 failed / 3 ignored**。三个 ignored 的外部模型/模板条件由模型 CI 显式执行相关路径，不能解释为从未验证。

上述 198 为 all-targets 的运行时测试计数；format 另有一项 mmap unsafe 契约的编译失败 doctest，通过后单独记录，不与运行时测试混计。

专项证据：

- format release 回归；两个带 ASan/覆盖率的 fuzz target 各 10,000 次本地运行；最新 CI 另执行 seeded smoke，范围见 [fuzz README](../../fuzz/README.md)。这些运行次数不能换算成安全证明。
- C golden reader 使用 OpenSSL SHA-256，并在 ASan/UBSan 下参与两种正例/四种反例的三语言比较，见 [conformance](../formats/qif-conformance.md)。
- 六个 64 MiB tensor/alias 组合在 48 MiB Linux 地址空间限制下验证；恶意重叠未知名称正常报错，见 [resource check](../../tests/conformance/qif_resource_bounds.py)。
- WASI-NN host feature 的压缩 builder 预算专项测试通过；Python benchmark 回归测试五项通过，包括真实驱动编排的合成检查器 fixture。合成 fixture 不算真实模型正确性证据。
- 两模型 F32/Q8_0/Q4_0 性能用同 runner 固定参考 `3cd5745…` 与当前代码各三轮交错测量，核对相同模型 hash、CPU、配置、输入与生成 token，比较各指标中位数。历史 RSS/KV 基线检查仍保留，原始样本随工作流上传。

远端同一代码基线四项工作流均成功：

| 工作流 | 结果与链接 |
| --- | --- |
| CI | [success](https://github.com/aiqubits/quantinfer/actions/runs/44113044413)；workspace、C/Python/Rust golden、Linux resource、instrumented fuzz、x86/ARM SIMD |
| Model Conformance | [success](https://github.com/aiqubits/quantinfer/actions/runs/44113044375)；两模型独立参考、六种基准变体、绑定/HTTP/WASM/WASI-NN 等已列步骤；自定义 external-artifact job 因本次 push 无该输入按配置 skipped |
| GGUF Interoperability | [success](https://github.com/aiqubits/quantinfer/actions/runs/44113044164) |
| WGPU Conformance | [success](https://github.com/aiqubits/quantinfer/actions/runs/44113044377) |

本次两个模型 job 的 CPU 均为 AMD EPYC 9V74，历史静态基线检查确实报告该 CPU
的 timing comparison unavailable；新增 paired 步骤仍实际执行并通过。下面是日志中
“当前代码中位数 / 同机固定参考中位数”的比率，按三位小数记录：

| 模型 | 变体 | Prefill | Decode latency | Decode throughput | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2 | F32 | 0.998 | 1.002 | 0.998 | 1.000 |
| SmolLM2 | GGML Q8_0 | 0.999 | 0.998 | 1.002 | 1.000 |
| SmolLM2 | GGML Q4_0 | 0.997 | 0.995 | 1.005 | 1.000 |
| Qwen3 | F32 | 0.983 | 0.992 | 1.008 | 1.000 |
| Qwen3 | Q8_0 | 0.999 | 1.001 | 0.999 | 1.000 |
| Qwen3 | Q4_0 | 1.001 | 0.998 | 1.001 | 1.000 |

来源为该 run 的 [SmolLM2 job](https://github.com/aiqubits/quantinfer/actions/runs/44113044375/job/134143530969)
和 [Qwen3 job](https://github.com/aiqubits/quantinfer/actions/runs/44113044375/job/134144718079)。
Prefill、TTFT、decode latency 比率上限为 1.35，decode throughput 下限为 0.70，
RSS 上限为 1.10；KV 使用量/容量要求整数精确一致。这些接近 1 的测量值用于证明
本轮未触发设定回归门槛，不用于声称已建立统计显著的性能提升。

厂商工作流成功也必须按其实际步骤解释：ABI/probe/mock 或编译成功不能等同于 CUDA/Metal/Ascend 实际设备的数值、吞吐和峰值内存验收。
