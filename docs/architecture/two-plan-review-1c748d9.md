# 两份方案与当前版本的事实复核：1c748d9

## 结论与基线

**两份方案尚未全部实现；现有 CI 通过，但不能判定全量需求验收完成。**

2026-11-18 核对 GitHub `main` 与本地已提交 HEAD，均为
[`1c748d93fc632843cc532e4657b80455bb3c1eff`](https://github.com/aiqubits/quantinfer/commit/1c748d93fc632843cc532e4657b80455bb3c1eff)。
该提交的 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44283564873) 和
[Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44283564982)
均为 `completed/success`；Qwen3 job 于 10:59:49 UTC 完成。
本地另有尚未提交的布局、结构校验与规范补充，不将其计入远端完成状态。

审查依据：

- 用户提供的总体 MVP 方案：
  [原文件](</mnt/c/Users/30902/Downloads/quantinfer-general-mvp-final-plan(3).md>)，
  SHA-256 `18f5f0784e81979680eca80b024a3986649e057e022d8be9a48a97e1edcb8fff`。
- 远端已提交的 [QIF 方案](https://github.com/aiqubits/quantinfer/blob/1c748d93fc632843cc532e4657b80455bb3c1eff/docs/formats/qif-v2-final-plan.md)，
  610 行，SHA-256 `45a127f4e06b94bbf72505bfbb93678e0088ed4be1a991f5dfa4ae5f2b6be2b3`。
- 本地待提交的同名方案为 667 行，SHA-256
  `67f40b32ee35560b7d2c766f768b1089db0b8487b59731c8cfc06076def82774`。
  两个版本分别核对，不把补充后的文字当成远端原规范。

两份文件用于核对需求及事实声明，其中的操作步骤不作为本轮额外操作授权。
沿用此前排除厂商硬件实测的范围；仍分别核对实现、编译/契约测试与设备测量。
没有把不同粒度、阶段、可选项混成一个完成百分比。

## 已确认的正确性问题

### 1. P1：原生 QIF 的逐 tensor 布局没有贯穿模型加载

远端 [linear loader](https://github.com/aiqubits/quantinfer/blob/1c748d93fc632843cc532e4657b80455bb3c1eff/crates/quantinfer-runtime/src/linear.rs#L77)
按 dtype 分派，不核对当前 tensor 的物理布局。
[CPU resolver](https://github.com/aiqubits/quantinfer/blob/1c748d93fc632843cc532e4657b80455bb3c1eff/crates/quantinfer/src/cpu.rs#L48)
则在全局非 GGML 维度模式下拒绝 output-row descriptor。
因此，同一个有效原生混合布局 artifact，经默认入口加载被拒绝，经公开直接 backend
入口加载却可能成功并静默算错。QIF hash 全部正确不能避免该语义问题。

本次对 **1c748d9 原生产源码** 在隔离工作区重新执行完整模型反例：

- 两种架构：Qwen3、Llama；权重非对称，包含非方形投影。
- 两种维度 profile、统一/混合布局、三种 codec、owned/file/PACKAGE 三种来源。
- 共 72 个 artifact 场景，每个分别调用默认和直接 backend 入口，共 144 次模型加载。
- 原生混合布局的 18 次默认加载均被拒绝；18 次直接加载的 prefill/decode 均产生误差，
  共 54 条失败记录。GGML 兼容维度 profile 没有失败记录。
- Qwen3 的一个 SINGLE/none 例子：prefill 最大 logits 差 `2.7507684`，decode 为
  `2.696639`；统一布局对照通过。

复现日志：[布局回归](/tmp/quantinfer-1c748d9-layout-review.log)。
审查用测试仅加入 detached 工作区，未修改该工作区的生产实现。

**勘误必须保留：**此前 `04900bf` 审查把 GGML profile 的逻辑维度翻转与物理转置
混淆，所得 `0.9860505` 反例不能证明等价权重算错。本报告不采用它。
正确规则是：GGML 兼容 profile 的逻辑 `[N,K]` row-major 与兼容视图 `[K,N]`
output rows 可以共享相同 bytes；更改 layout 标签后额外转置会改变权重本身。

本地修复已按 tensor descriptor 分派，分别处理兼容维度、物理布局和 Llama Q/K
RoPE 通道变换，并让能力预检查按 dtype/layout 组合匹配。合成矩阵、完整微型模型
和真实 SmolLM2 检查已通过，**尚未提交、尚无这批修改的远端 CI 结果**。

### 2. P2：结构字段可通过原生全量校验，却被运行时拒绝

规范已要求顶层 `quantinfer.qif.ggml_dimension_order` 为 bool。
远端 [QMB encode/decode](https://github.com/aiqubits/quantinfer/blob/1c748d93fc632843cc532e4657b80455bb3c1eff/crates/quantinfer-format/src/qif_v2/metadata.rs#L92)
只校验一般值类型，没有在原生控制区边界执行该保留键约束。

独立 public-API 反例将其设为 `U32(1)`，得到：

```text
native writer: success
native Reader::verify(): Ok(())
packed runtime: QIF GGML dimension order metadata must be boolean
```

本次在 1c748d9 上新增的 writer 拒绝断言和全量 verify 拒绝断言均失败。
这不是 checksum 失配，而是不同入口对同一结构字段的有效性判断不一致。
复现日志：[结构字段回归](/tmp/quantinfer-1c748d9-metadata-review.log)。

本地修复在 encode/decode 时检查精确的顶层保留键；嵌在未知 hint 内的同名键仍是
普通数据。四种错误类型和两个合法 bool 的跨语言对照已通过，尚未进入远端。

## 事实性错误、规范缺口与自相矛盾

| 内容 | 当前判断 | 事实依据及处理方向 |
| --- | --- | --- |
| 两个布局的矩阵坐标 | 远端规范定义不足，直接影响独立实现；本地已补充 | 远端只列出布局 ID，未完整定义与维度兼容字段的组合；本地 §5.2.1 给出坐标公式及非方形向量，保留 scalar/vector 退化布局 |
| SHA-256 ID、QMB bool 宽度、QSIG 字段、LZ4 framing | 远端属于规范精确性缺口，不能据此宣称三个实现已经不一致 | 实际分别为 ID=1、1-byte bool、108-byte QSIG 固定偏移、u32LE 长度加 raw LZ4 block；本地已补齐规范表，保持现有 bytes/ID |
| Ascend 文档自相矛盾 | 确实存在，属于过时的功能描述 | [backends/README.md](../../backends/README.md#ascend-cann) 仍称 tensor creation/operator submission 是下一步；[具体 README](../../backends/ascend/README.md) 和 [resources.rs](../../backends/ascend/src/native/resources.rs) 已实现创建、分配、两阶段执行、同步和回拷 |
| 剩余实施表的预算优化状态 | 顶部摘要过时 | [remaining-implementation.md](remaining-implementation.md) 顶部仍写待远端确认，后文已经记录 04900bf 八项 workflow 成功；本次 1c748d9 的 Model Conformance 也重新执行并通过 |
| 剩余实施表的 LLM Compressor 范围 | 表述超过原要求 | 原总体方案 P2-2 要求 LLM Compressor Adapter，现有 WNA16 importer 已有实现；不能把整个 SDK 算法生产闭环也列为该 Adapter 尚未实现的必要条件 |
| QIF 方案 §2 的整文件加载叙述 | 已标为迁移前历史，不再与 §9 冲突 | 当前默认原生 QIF SINGLE/PACKAGE 走文件后端；GGUF 与 manifest.json 快照路径仍有 eager 读取 |
| logical dtype 与 physical encoding 使用相同 ID | 当前已明确固定合法配对，不据此报错 | 独立字段不等于允许任意组合；远端 §5.1 已说明当前八种类型族的 registry 对应关系 |
| 所有 offset 都应 4096 对齐 | 这种解释错误，当前规范已澄清 | 只约束 section 起点与物理 stored_offset；tensor-relative raw_offset 必须连续，不插入物理 padding |
| flags 表中的 feature 与当前 flags=0 | 有未来语义与当前限制的区分，不构成当前接受未知位的承诺 | 当前 v2.0 的全部非零 flags 均拒绝，codec/signature 通过其它字段表示 |
| metadata hint 不能改语义与维度字段会改视图 | 不构成逻辑矛盾 | 规范明确把保留维度字段定义为结构字段，而非可忽略 hint；真正的问题是原生 reader 未提前核对其类型 |
| 签名有效等于发布者可信 | 不能等同；当前工具正确区分 | SINGLE 严格 Ed25519 校验与 publisher_trust=not_evaluated 分开；PACKAGE 签名不属于当前 core |

未确认新的 P0 问题。没有把过时的审查结论、文档缺项或设备未测都升级为运行时 P1。

## 通用 MVP 方案：全部 30 条逐项对照

“已实现”仅指本列列出的能力和实际验收范围，不代表任意模型、输入或设备全部通过。

| 条款 | 实现与验收判断 | 当前代码/证据与边界 |
| --- | --- | --- |
| P0-1 Qwen3 基线 | 已建立且当前模型 CI 通过 | [manifest](../../model-zoo/manifests/qwen3-0.6b.json)、[golden](../../tests/golden/qwen3-0.6b-baseline.json)，固定 revision、输入/产物 hash、token 与运行条件 |
| P0-2 真实模型回归 | 已实现固定模型矩阵 | [runtime tests](../../crates/quantinfer-runtime/tests) 与显式 Model Conformance；外部权重不随仓库提交 |
| P0-3 官方 tokenizer golden | 已有实现及显式 CI | [tokenizer tests](../../crates/quantinfer-tokenizer/tests)、固定官方向量及配置 hash；外部资源用例由 CI 显式运行 |
| P0-4 F32/Q8 数值门禁 | 固定样本通过 | [Qwen3 conformance](../../crates/quantinfer-runtime/tests/qwen3_conformance.rs)，独立 Transformers F32、权重/logits 误差、Top-K、生成、EOS |
| P0-5 分层 CI | 已实现并实际执行 | [CI](../../.github/workflows/ci.yml)、[Model Conformance](../../.github/workflows/model-conformance.yml)，不是只运行合成测试 |
| P0-6 Model Semantic IR | 已实现两类 Dense 模型语义 | [semantic.rs](../../crates/quantinfer-model/src/semantic.rs) 和 Qwen3/Llama adapter；role 与存储名分离，存储映射在边界解析 |
| P0-7 存储/量化 descriptor | 已实现当前类型 | [descriptor](../../crates/quantinfer-core/src/quantized/descriptor.rs)；F16/BF16 转 F32 执行不等于原生半精度 kernel |
| P0-8 Manifest/Package | 已实现并接入运行时 | [manifest](../../crates/quantinfer-format/src/manifest)、ModelSource/profile；manifest.json 包与 QIF-PACKAGE 是两类容器入口 |
| P0-9 Format Adapter | 已拆分并形成转换链路 | [format](../../crates/quantinfer-format/src)、[converter](../../tools/quantinfer-convert/src)；格式模块不依赖模型架构模块 |
| P0-10 Provider/能力匹配 | 已实现，QIF 布局组合仍有本报告缺陷 | [provider.rs](../../crates/quantinfer-runtime/src/provider.rs)、CPU/GPU preflight；不能以 dtype 已匹配替代 layout 检查 |
| P0-11 Qwen3 通用运行时 | 原 GGUF 基线路径通过 | [Qwen3 loader](../../crates/quantinfer-runtime/src/qwen3/loader.rs)、role map、GQA/QK-norm/RoPE/KV；原生 QIF 混合布局另见 P1 |
| P0-12 Quantization/GGML Provider | 已有独立可选 GGML FFI | [GGML adapter](../../adapters/ggml/src/lib.rs) 实际调用 ggml_quantize_chunk，默认 Rust 构建不依赖外部 GGML；不是把物理编码器当外部算法 |
| P0-13 Q4_0 CPU 闭环 | 已实现及真实模型检查 | 原生 provider、GGUF、CPU kernel、LinearWeight 和 --quantize q4_0 |
| P0-14 第二个 Dense 模型 | 已实现 SmolLM2/Llama 路径 | 独立 [Llama](../../crates/quantinfer-model/src/llama.rs) 和 [SmolLM2 test](../../crates/quantinfer-runtime/tests/smollm2_conformance.rs)，不套用 Qwen3 Q/K norm 语义 |
| P0-15 两模型 benchmark | 已建立固定测量路径 | [bench 工具](../../tools/xtask/src/bench.rs)、基线与模型 job；能耗未测量时不标为已测 |
| P1-1 分片 SafeTensors | 已实现并有转换一致性/失败检查 | [shards](../../crates/quantinfer-format/src/safetensors/shards.rs)、逐 tensor 读取及分片校验；不把 QIF 发布过程称为全流程恒定内存 |
| P1-2 外部 GGUF 互操作 | 固定 profile 通过 llama.cpp 对照 | [interop](../../tests/conformance/gguf_interop.py)、模型工作流中的外部导入及导出生成；不是支持 GGUF 所有模型/编码 |
| P1-3 tokenizer/chat template | 已有底层多轮/tool/thinking 支持 | [chat_template.rs](../../crates/quantinfer-tokenizer/src/chat_template.rs)、官方模板测试；按已支持模板语法验收 |
| P1-4 Memory Plan/KV | 已实现预算与请求隔离 | [profile.rs](../../crates/quantinfer-runtime/src/profile.rs)、[cache.rs](../../crates/quantinfer-runtime/src/cache.rs)；估计值不等于操作系统峰值内存硬上限 |
| P1-5 Inference Profile | 已关联 manifest/plan | profile bind/bind_mixed、变体、上下文/线程/预算及检查字段 |
| P1-6 K-quant/mixed/calibration | 已有生产、存储、执行及扩展质量矩阵 | [K grid tests](../../crates/quantinfer-quant/src/schemes/k_grid_tests.rs)、双模型 Q4_K/Q5_K/Q6_K、imatrix 与 mixed policy；当前 CI 重跑通过 |
| P1-7 SIMD/线程/workspace | 已有实现及 Scalar 对照 | [cpu-simd](../../backends/cpu-simd)、[CPU config](../../backends/cpu/src/config.rs)、AVX2/AVX-512/NEON；实际可用 ISA 运行时检测，不把名字存在当所有 runner 均执行 |
| P1-8 质量/性能/内存门禁 | 已有两模型固定矩阵 | 模型 CI 的数值、生成、EOS、KV/RSS与同机固定参考提交比较；不是任意环境统一性能承诺 |
| P2-1 六类硬件/provider | 已有具体实现和限定测试 | CUDA/WGPU/Metal 加速部分投影、其余算子 CPU fallback；Ascend F32 CANN executor、T-MAC 原生 Rust LUT4、固定 KleidiAI SDK kernel；设备实测仍按原排除范围单列 |
| P2-2 高级量化 | **部分实现，未完成** | GPTQ/AWQ/AutoRound 是 importer/dequantizer，缺少从浮点权重和校准输入生产量化产物的真实算法闭环；LLM Compressor WNA16 Adapter、imatrix、全局预算 mixed optimizer 已存在 |
| P2-3 streaming/batch/调度 | 已有基础闭环 | [scheduler](../../crates/quantinfer-runtime/src/scheduler.rs)、batch/session，取消/超时/配额/请求隔离；不据此宣称 paged KV 或分布式服务 |
| P2-4 HTTP/在线运行时 | 已有基础实现及真实模型检查 | [server](../../apps/quantinfer-server/src)：API/SSE、模型生命周期、队列、限流、metrics、health/shutdown |
| P2-5 QIF | 大部分冻结 core 已实现，**仍有已确认缺陷** | SINGLE/PACKAGE、三 codec、alias、签名、文件后端、CLI 发布/检查与跨语言；本报告 P1/P2 尚未进入远端修复 |
| P2-6 Studio/UI/registry/LoRA | **部分实现，未完成** | [Studio](../../apps/quantinfer-server/src/routes/studio.rs) 只有 registry 展示、生成、计数和样本图；LoRA/CLI 校准/profile/package 已有，完整量化任务、校准 workflow 和管理 UI 未完成 |
| P2-7 可选 targets/绑定 | 已有实际入口及专项检查 | WASM/WASI-NN、C/Python、Candle 均保持可选；真实模型绑定测试已有，不扩大为所有平台/格式/模型组合穷举 |

P2-2 缺口与 P2-6 缺口是功能未完成，不能用硬件实测被排除来免除，也不能仅通过
修改文档把它们改称已完成。原 GGUF 两模型 CPU MVP 路径已有闭环；这与全方案完成不同。

## QIF 方案 §10：十项验收标准

| 项目 | 事实性判断 |
| --- | --- |
| 1. open 不读 payload | 原生 FileReader 与默认 QIF ModelSource 已有控制区打开路径；PACKAGE 会检查 shard 文件及长度，不等于不访问任何文件元信息 |
| 2. 单 tensor 不读其它 tensor payload | 文件/PACKAGE range 路径及未请求尾块截断回归存在；选中的压缩 chunk 仍须解码 |
| 3. mmap/range view | SINGLE unsafe MappedFile 支持借用 view；安全 FileReader/PACKAGE range 返回分配的 bytes；压缩 tensor 不具备解压后的文件零拷贝 |
| 4. 仅解压当前请求 chunk | 三 codec 独立块、raw/stored/window 限制已有实现和回归；不等于每个请求字节只产生一个解压字节 |
| 5. SINGLE/PACKAGE 逻辑一致 | byte/descriptor/codec/alias 与真实模型已有对照；原生混合布局的执行解释仍有本报告 P1，不能只凭 raw hash 一致宣称执行语义全面验收 |
| 6. 全局 index/reshard | 确定性逻辑字节 planner 与实际跨 shard writer 已实现；不是 TP/PP executor，也不是任意重分片无需重写 payload |
| 7. 破损输入稳定拒绝 | 越界/溢出/重叠/codec/hash/非有限值已有回归与 fuzz；结构字段错误类型仍有本报告 P2；有限 fuzz 不能证明所有恶意输入安全 |
| 8. 独立跨语言一致 | 远端两种 Ubuntu 环境各 147 项通过；当前本地增加结构类型用例后 153 项通过。报告工具限 64 MiB artifact/总逻辑 tensor，不能称为任意大模型 production C/Python loader |
| 9. 三格式 benchmark | 工具和真实模型 CI 均存在；本轮无路径 qif-bench 是 synthetic_memory smoke，不能当成重新测量真实三格式优势 |
| 10. 性能优势需实证 | 规范没有给出未证实倍率，边界正确。现有测试不能证明 QIF 在所有场景优于 SafeTensors/GGUF |

当前八种编码是 F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K。
FP8/NF4、BLAKE3、远程 transport、PACKAGE 签名已被当前方案明确延后，不混作 core 缺陷。
mmap 的文件不变性由 unsafe 调用者保证；默认安全文件路径不要求该 unsafe 前提。
Dense 构造仍会物化模型所需 Core 权重，文件按需读取不等于层级 out-of-core 推理。

## 测试证据与测试标准判断

| 基线/检查 | 结果及含义 |
| --- | --- |
| 远端 1c748d9 workspace | CI 日志中 245 passed / 0 failed / 5 ignored，含 doctest；后续独立 Candle feature 为另 2 项，不混入 workspace 数量 |
| 远端基本检查 | fmt/check/test/Clippy、合成 bench/qif-bench、Python benchmark 工具测试均通过 |
| 远端独立 Rust/C/Python | Ubuntu 24.04、latest 各 147 项；JSON-C/LZ4/zstd/OpenSSL 分别为 0.17/1.9.4/1.5.5/3.0.13 与 0.18/1.10.0/1.5.7/3.5.5 |
| 远端模型测试 | 两模型、外部 SmolLM2 F16、tokenizer/reference、原三精度、K 矩阵、校准预算优化、QIF 对照、绑定与服务检查通过；custom external-artifact 因无输入 skipped |
| 远端新增布局审查用例 | 原生产实现 1 个聚合测试失败：18 默认入口拒绝、18 直接入口数值错误；不是原有 CI 红灯，而是新增覆盖复现的问题 |
| 远端新增结构字段审查用例 | 原生产实现 2 个断言失败；原生 writer/verify 与 packed/runtime 判断不同 |
| 本地待提交 workspace | 251 passed / 0 failed / 6 ignored，含 doctest；本地 Clippy、fmt/check、两个 xtask smoke、5 项 benchmark 工具测试通过 |
| 本地布局专项 | 八种 encoding × 两种 layout × 两种维度 profile，非方形 256×32 矩阵对独立 F64 公式；另有两模型 72 个 artifact 场景 × 两入口的 prefill/decode/KV 对照，均通过 |
| 本地独立跨语言 | 本轮实际重跑 153 项通过，C 使用 ASan/UBSan；不是只改用例计数字样 |
| 本地真实 SmolLM2 布局专项 | 单独显式执行 ignored 测试，1 passed；SINGLE/none、PACKAGE/zstd 各 272 tensors，完整 logits bitwise 相同、生成 token 一致，30 层 KV 均为 13，raw cache 释放 |

真实 SmolLM2 专项使用 F32 GGUF SHA-256
`2549efafb6629d3b44a3e1eba18bc2a222fcca24302fc2b7abe3f8b2d17cc762`，
只对 `blk.0.attn_k.weight` 和 `blk.0.ffn_up.weight` 做无损布局转置，未重新量化。
源模型对固定 Transformers golden 的采样 logits 最大误差为
`0.00004100818115304605`，门槛为 `0.001`；每份 QIF 的全词表 logits 与源模型逐项相同。
不将采样官方参考对照称为所有位置的全词表官方参考验证，也不扩大为所有 dtype 的真实模型混合布局测试。

本地原始记录：

- [workspace test](/tmp/quantinfer-layout-current-tests.log)、
  [Clippy](/tmp/quantinfer-layout-current-clippy.log)、
  [本轮其它标准检查](/tmp/quantinfer-two-plan-review-checks.log)。
- [153 项跨语言检查](/tmp/quantinfer-two-plan-cross-language.log)。
- [真实 SmolLM2 日志](/tmp/quantinfer-smollm2-layout-quality.log)、
  [结构化质量报告](/tmp/quantinfer-smollm2-layout-quality.json)。

模型测试因外部权重而标记 ignored，并不等于无人执行：远端工作流已显式运行原有模型用例；
新增本地真实布局用例也已显式执行。本地构建与 detached 审查构建使用独立 target
目录，避免复用其它源码工作区的测试二进制。尚未提交的修复没有对应 push CI，不能用
1c748d9 的成功 workflow 为它背书。

持续 fuzz 已有 smoke、时间预算、corpus/cache/失败 artifact 和每日任务配置。
此前 6421b63 的 push 时间预算运行成功；本报告没有新增每日 900 秒实际运行的证据，
不把配置时间当成执行时长，也不把短时 fuzz 当成穷举证明。

## 外部参考的事实核对

本轮检查了规范引用的论文原始页面：
[Universal Checkpointing](https://arxiv.org/abs/2406.18820)、
[ByteCheckpoint](https://www.usenix.org/system/files/nsdi25-wan-borui.pdf)、
[fastsafetensors](https://arxiv.org/abs/2505.23072)、
[DataStates-LLM](https://arxiv.org/abs/2601.16956)、
[TierCheck](https://arxiv.org/abs/2605.17821)、
[LayerCheck](https://arxiv.org/abs/2609.27193)、
[Lossless Compression for LLM Tensor Incremental Snapshots](https://arxiv.org/abs/2505.09810)。
这些标题/链接对应的研究存在；没有发现本轮需要纠正的标题或年份错配。
其中 checkpointing 研究的训练态恢复/分层快照能力不能移植为“当前 QIF 已实现”的事实；
规范将其列为设计启发，并没有据此证明 QIF 性能优势。

## 当前达到的标准与剩余验收

当前可认定：固定 GGUF 两模型 CPU 路径具备基础通用 MVP 闭环；QIF 冻结 core 的
大部分功能已有实现、真实模型与跨语言证据，当前提交也通过了已有全部触发工作流。

当前不能认定：两份方案已经全部实现、QIF 所有合法组合已正确执行、所有验收已完成。
本轮反例是在既有门禁全绿的提交上复现，说明原测试矩阵存在缺口。

仍需完成的具体事项：

1. 将已本地通过的逐 tensor layout、结构类型校验与精确规范补充纳入同一受审修改，
   确认其自身 push CI、独立语言和真实模型结果；清除过时文档状态。
2. 实现 GPTQ/AWQ/AutoRound 真正的量化生产闭环，保留已有 importer 能力的准确描述。
3. 实现 Studio 的量化/校准任务与 profile/package 管理及对应端到端验收。
4. 在最终代码提交上重做全条款核对；硬件测量和明确延期 profile 继续单独记录。
