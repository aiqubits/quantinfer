# 两份方案与最新代码的事实复核（2026-11-18）

## 审查对象与结论

远端 `main` 与本地已提交 HEAD 均为
[`e1a1c96dbc2f25bcafb9200fe49d511b27c435ae`](https://github.com/aiqubits/quantinfer/commit/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae)。
本报告编写前再次查询远端，未发现更新提交。

对照的需求文件：

- 用户提供的《QuantInfer 通用量化推理框架 MVP 最终实施方案》，SHA-256
  `18f5f0784e81979680eca80b024a3986649e057e022d8be9a48a97e1edcb8fff`。
- 该提交的 [QIF v2 最终方案][qif-plan]，SHA-256
  `a5520eea0873a6bbbe12118a696ed2dcc87e0e6e3662aebd718a3cf6d15f7b24`。

**不能认定两份方案已全部实现、没有事实或语义问题、全部验收完成。**

通用方案的 P0 主链路和大部分 P1 能力已有实现，并通过固定模型的现有门禁。
QIF 当前冻结的八种编码、两种容器 profile、三种 codec、SINGLE 签名、文件按需读取、
默认运行时接入和独立跨语言验证已有执行代码及测试。仍有实际的 K-quant 生产端问题、
高级量化与 Studio 功能缺口，以及规范文字需要澄清的地方。

本次没有确认新的 P0/P1 缺陷；确认了应修复的 K-quant P2 问题。这一结论只覆盖本次
审查与测试，不是所有输入均无缺陷的证明。继续遵守此前排除厂商设备实测的范围；
没有把“缺少设备测量”计为本次必须补齐的功能缺口。

本地起始时已有四个 K-quant 源文件修改、一个新量化测试文件和一个新运行时测试文件。
它们未提交、未推送，本报告不将这些内容计入远端完成状态。测试也分别对固定 HEAD
的独立工作区与当前本地修改执行。

## 已复现的问题与事实边界

### 1. P2：K-quant 选码使用的参数与最终存储码表不一致

远端 [Q4_K][q4k]、[Q5_K][q5k]、[Q6_K][q6k] 的生产顺序是：

1. 从每个子块计算浮点 scale/min，并立即选择整数码。
2. 再把子块 scale/min 量化成 6-bit 或 i8 系数。
3. Core block 保留 F32 全局 scale；[GGUF writer][gguf-writer] 随后将它写成 F16。

整数码没有按第二、三步后的实际码表重新选择，因而产生可避免的重建误差。
在不修改远端生产代码的独立工作区，加入独立标量枚举对照后，三种编码均复现失败：

| 编码 | 一个输入值 | 当前码的绝对误差 | 同一最终码表中最优码的绝对误差 |
| --- | ---: | ---: | ---: |
| Q4_K | -0.0036300002 | 0.0012441101 | 0.0004177867 |
| Q5_K | -0.011559473 | 0.0004546773 | 0.0003489126 |
| Q6_K | -0.011559473 | 0.0005043875 | 0.0003573168 |

这里枚举的是已经确定的存储码表中的全部合法整数码，不是把另一种量化算法的结果
强行作为正确答案。它也不证明逐元素误差减小会让每个模型的 logits 或生成质量改善。

现有本地补丁已将选码移到最终存储参数确定之后，并处理 F16 表示范围；六项专项
测试通过。但补丁尚未进入远端，扩展模型验收仍未闭环，不能写成远端已解决。

补充区别：六项新测试在原生产代码上得到 2 passed / 4 failed。其中三项是上表的
选码反例；第四项要求量化阶段提前拒绝不能以有限 F16 导出的 scale。后者属于新增
边界要求。现有 GGUF writer 已会在 finish 阶段拒绝这种溢出，不能将第四项失败
描述成“当前 writer 已成功发布 Inf 权重”。

### 2. 内存误差报告不能直接代表导出文件误差

使用公共 API，对 32 个 `1e-8` 的 F32 值分别做原生 Q8_0 和 Q4_0：

| 编码 | 对内存 QuantizedData 的 relative L2 | 写 GGUF 并重新读取后的 relative L2 |
| --- | ---: | ---: |
| Q8_0 | 0 | 1 |
| Q4_0 | 0 | 1 |

原因是内存 block 的 F32 scale 在 F16 导出时下溢为零。
[error_report][quant-provider] 对传入内存对象的计算本身正确；它没有接收导出文件，
不应称其计算公式错误。此反例说明，产物质量验收必须对实际写出的 bytes 重读后计算，
或明确限制为已经实现存储精度的量化对象，不能拿导出前报告替代产物验收。

已有真实模型 conformance 使用重读的 GGUF，不受“只测内存对象”的误解影响；本次
没有据此宣称已有模型门禁被绕过。Core 还允许非 F16 的 F32 scale 使用场景，不宜
为修复导出边界而无条件改变所有 Core block 的数值语义。

### 3. 方案中的旧加载现状与新加载现状冲突

[QIF 方案 §2][qif-old-state] 仍写 `ModelSource::open_reader` “当前先读取整个文件”。
实际 [ModelSource][model-source] 的原生 QIF 路径已走
`open_path_with_memory_budget`，由 [file backing][backing] 打开控制区、按 tensor 读取。
方案 §9 已正确描述新行为。

§2 的相关表述应明确标为“废弃 v1 时的迁移前快照”。`manifest.json` 模型包的
已校验 owned snapshot 和 GGUF eager 路径仍存在，不能因此说所有入口都已惰性读取。

### 4. 对外 reshard API 名称不一致

[QIF 方案 §10.6][qif-acceptance] 使用 `plan_qif_v2_reshard` 作为 API 名称；
[实际公开出口][qif-api] 是 `quantinfer_format::qif::plan_qif_reshard`。
前者属于私有 wire module 内部实现名。对外用法应写公开名称，内部实现可以保留原名。

### 5. 对齐条款应明确限定为物理存储偏移

方案 §6.1 写“section 和 chunk offset MUST 是 4096 字节的倍数”，但 chunk 同时有
`raw_offset` 与 `stored_offset`。实际 [PACKAGE writer][package-writer] 只对物理
`stored_offset` 做 4096 对齐，`raw_offset` 是连续的 tensor 逻辑字节位置。

本次用 320-byte F32 payload、`max_shard_bytes=100` 生成并成功全量 verify 的 PACKAGE：

```text
raw_offsets    = [0, 100, 200, 300]
stored_offsets = [0,   0,   0,   0]
```

应将规范写成“文件 section 起点及 chunk 的物理 stored_offset 对齐；逻辑 raw_offset
不受此对齐约束”。这可以澄清现有 wire，无需错误地给逻辑数据插入 4096 padding。

### 6. logical dtype 与 physical encoding 的独立性尚需定清语义

方案 §5 要求分开记录 `logical_dtype`、`physical_encoding`、量化方案和 layout，
并称 Q4_K 不应同时充当 dtype、量化规则和物理布局名称。当前确实有四个独立字段，
但 [writer][descriptor-ids] 对前两项写入同一 dtype ID，
[reader][descriptor-validation] 强制两者同时等于 descriptor 的编码类型。
本次生成并 verify 的 Q4_K fixture 为：

```text
dtype=5, physical_encoding=5, quantization_scheme=3, layout=1
```

仅仅两个 ID 数值相等不构成错误；关键是当前两字段的合法组合被强绑定，且 Core
descriptor 没有单独的 logical element dtype。现有实现可确定地读写这八种编码，
但不能称为已实现任意独立的逻辑 dtype / 物理编码组合。

这是规范语义和可扩展性边界，不是本次发现的 payload 读取错误。应补充两个字段
各自的 registry 定义及合法组合表，明确 v2.0 的固定映射；如果后续确需独立逻辑
元素类型，应单独演进 profile，不能静默重新解释已冻结的 ID。

## 通用方案逐项对照

“已有实现”不等于所有模型、输入、ISA 和设备都已验收。没有使用缺乏共同分母的完成百分比。

| 条款 | 当前判断 | 代码与验收边界 |
| --- | --- | --- |
| P0-1 固定 Qwen3 基线 | 已有实现及固定证据 | [manifests][model-manifests]、revision/hash、golden 与模型工作流 |
| P0-2 真实模型回归 | 已有实现并实际执行 | [runtime tests][model-tests] 与两模型 job；根目录资源由显式入口使用 |
| P0-3 官方 Tokenizer golden | 已有实现及 CI | [tokenizer tests][tokenizer-tests]，包含配置与模板的显式 ignored 执行 |
| P0-4 F32/Q8 数值门禁 | 固定模型范围通过 | 独立 Transformers F32、量化权重、logits、Top-K、生成及 EOS |
| P0-5 Conformance CI | 已实现 | [模型工作流][model-workflow] 与快速、互操作检查分层 |
| P0-6 Semantic IR | 已实现 Dense 范围 | [ModelSpec/semantic roles][semantic]，Qwen3/Llama adapter |
| P0-7 Storage/Quantized Descriptor | 已实现当前固定编码 | [descriptor][descriptor]；F16/BF16 的 CPU 执行仍转 F32，符合方案边界 |
| P0-8 Manifest/Package | 已实现 | [manifest][manifest] 与 ModelSource；普通模型包和原生 QIF PACKAGE 分别处理 |
| P0-9 Format Adapter | 已实现主链路 | SafeTensors/GGUF、adapter、显式 layout、可选量化、writer；QIF 发布仍复用 GGUF 语义中间层 |
| P0-10 Provider/Matcher | 已实现 | [provider/resolver][runtime-provider]、[kernel contract][kernel] 与 requirements 检查 |
| P0-11 通用 Runtime | 已实现 Qwen3/Llama Dense | [runtime][runtime]，没有据此宣称任意架构或 MoE 支持 |
| P0-12 Quantization/GGML Provider | 已实现 | [provider][quant-provider] 与可选固定版本 [GGML FFI][ggml]；真实 GGML 调用与编码兼容分开 |
| P0-13 原生 Q4_0 | 已实现固定模型闭环 | descriptor、quantizer、GGUF IO、CPU dispatch、CLI、质量门禁 |
| P0-14 第二 Dense 模型 | 已实现 | SmolLM2/Llama 的配置、adapter、独立 golden、执行及批处理对照 |
| P0-15 真实 benchmark | 已实现两模型三变体 | [benchmarks][benches]；能耗明确未测 |
| P1-1 分片 SafeTensors | 已实现并有转换回归 | [shards][shards]，按 tensor 定位读取并保留多文件校验 |
| P1-2 GGUF 互操作 | 已实现已列矩阵 | [互操作工作流][interop]，固定 llama.cpp 与 metadata/layout/token 对照 |
| P1-3 官方模板语义 | 已实现已列用例 | system、多轮、assistant、tools、thinking 等 golden；不是任意 Jinja 执行器 |
| P1-4 Memory Plan/KV | 已实现估算与预检 | [KV][cache] 及 profile；不是 allocator/解码器/运行状态全部峰值的硬上限 |
| P1-5 Inference Profile | 已实现 | [profile][profile] 关联 variant/provider/context/threads/budget 等声明 |
| P1-6 K-quant/mixed/calibration | 有实现，仍有本报告的缺陷及质量覆盖缺口 | 三种 K IO/kernel 已有；mixed 为分配/分组规则；imatrix 有实际激活采集；完整 K 变体质量矩阵未闭环 |
| P1-7 CPU 性能实现 | 已实现可选 SIMD/线程边界 | [cpu-simd][simd] 含 AVX2/AVX-512/NEON；测试会跳过不支持的 ISA，不将一个 runner 成功扩大为全部 ISA 都执行 |
| P1-8 质量/性能/内存门禁 | 已实现当前两模型矩阵 | F32/Q8_0/Q4_0 的实际质量、RSS/KV 与同机配对计时；不覆盖任意新格式组合 |
| P2-1 厂商 kernel | 已有具体实现和限定 profile | CUDA/Metal/WGPU；Ascend CANN F32 executor；原生 Rust T-MAC LUT4；固定 SDK KleidiAI。后两者不是上游全量工具栈或全模型性能结论 |
| P2-2 高级量化 | 未全部实现 | [GPTQ/AWQ/AutoRound/LLM Compressor adapters][adapters] 有已有产物导入和反量化；前三者没有浮点权重加校准输入到量化产物的算法闭环；LLM Compressor 已有其限定 import adapter；复杂 mixed 优化尚无实现 |
| P2-3 Streaming/Batch/调度 | 已有基础闭环 | [scheduler][scheduler] 与 batched forward，包含准入、取消、超时、KV 配额和隔离测试；不是 paged KV 或分布式调度 |
| P2-4 HTTP Server | 已有基础闭环 | [server][server] 的 API/SSE、模型生命周期、队列、限流、metrics、health/shutdown 与真实模型检查 |
| P2-5 QIF | 当前 core 已实现并有验收证据 | 见下表；不是比较性能优势的证明 |
| P2-6 Studio/Registry/Adapter | 部分实现 | [Studio][studio] 是 registry 展示、生成和计数/样本图表；LoRA、registry、CLI 校准已有；完整量化任务、校准 workflow、profile/package 编辑管理与任务生命周期未完成 |
| P2-7 可选 target/绑定 | 已有实际入口及专项检查 | WASM、WASI-NN、C/Python、Candle；按已执行场景验收，不扩大为所有目标平台和全模型的组合穷举 |

## QIF 方案 §10 验收逐项对照

| 条款 | 判断与证据 |
| --- | --- |
| 1 open 不读 payload | FileReader 已实现；原生 QIF 的默认 ModelSource 也已接入，不再是历史报告中的缺口 |
| 2 单 tensor 请求不读其它 tensor payload | 文件/PACKAGE range reader 和未请求尾块截断测试存在；同一压缩 chunk 内仍须解码该 chunk |
| 3 无压缩 mmap/range view | SINGLE 的 unsafe MappedFile 加 borrowed view 已实现；安全 FileReader/PACKAGE range 返回分配的 bytes，不等于所有入口都是零拷贝 |
| 4 仅解压请求 chunk | none/zstd/LZ4 独立块；raw/stored/window 资源限制与回归存在；不是读取一个字节只解压一个字节 |
| 5 SINGLE/PACKAGE 逻辑一致 | Rust 固定编码、codec、alias、跨 shard 对照及独立语言矩阵；真实 SmolLM2 Q4_0 六组合也已执行 |
| 6 全局 index / reshard | [planner][reshard] 和跨 shard writer 有实现；是逻辑字节放置与最终物理布局，不是 TP/PP 分布式执行器或无需重写 bytes 的任意重分片 |
| 7 破损输入稳定拒绝 | 越界、溢出、重叠、错误 codec、hash、有限值和资源边界有回归与 ASan fuzz；有限输入集不能证明所有恶意输入安全 |
| 8 Rust/C/Python 一致 | 两个独立跨语言 job 各 147 项成功；八种编码、两种 layout、QMB、三种 codec、alias、PACKAGE、SINGLE Ed25519 与工具的可信公钥策略。工具有 64 MiB artifact/总逻辑 tensor 报告限制，不能称为任意大模型 C/Python production loader |
| 9 三格式 benchmark | 工具已有并在 SmolLM2 CI 实际执行；三进程、对应 embedding、4096-byte range、独立 QIF verify。源 SafeTensors 是 BF16，转换产物 embedding 是 F32；缓存未控制，不能把耗时直接归为格式的固有优劣 |
| 10 性能优势实证边界 | 文档正确保留为待证目标；当前结果没有建立 QIF 普遍优于 SafeTensors/GGUF 的结论 |

安全与部署边界：[签名实现][signature] 验证嵌入公钥的数学有效性；CLI 明确报告
`publisher_trust=not_evaluated`，没有把自带公钥视为可信发布者。PACKAGE 签名、BLAKE3、
FP8/NF4、新 encoding、远程 transport 属于当前方案明确延后项，不计作冻结 core 未实现。
Dense 构造仍解码并驻留所需权重，file backing 不代表层级 out-of-core 推理。

持续 fuzz 已有三 target、corpus、失败 artifact、push 时间预算及每日 900 秒配置。
最新主 CI 的 smoke 和此前 push 的时间预算运行均有成功证据；本次查询可见的
schedule 历史未提供 QIF 每日 900 秒实际运行的证据，因此没有把配置值写成已完成的
运行时长，更没有称其为安全穷举。

## 本次测试证据

固定 HEAD 的独立工作区先执行标准检查，之后才添加审查用临时测试。原标准检查结果：

| 检查 | 结果 |
| --- | --- |
| `cargo fmt --all -- --check` | 通过 |
| `cargo check --workspace --locked` | 通过 |
| `cargo test --workspace --locked` | 217 项运行时/单元/集成测试通过，3 ignored；另 1 项 compile-fail doctest 通过，合计 218 passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | 通过 |
| `cargo run --locked -p quantinfer-xtask -- bench` | 通过；合成 kernel smoke |
| `cargo run --locked -p quantinfer-xtask -- qif-bench` | 通过；无路径参数，因此只是 `synthetic_memory` smoke |
| Python benchmark 工具测试 | 5 项通过 |

审查过程中加入的 K 码表测试在远端原生产实现上复现前述失败。当前含未提交补丁的
本地工作区另测得到 224 项运行时/单元/集成测试通过、4 ignored，另 1 项 doctest
通过，合计 **225 passed / 0 failed / 4 ignored**；fmt 和 all-targets Clippy 通过。
新增 ignored 的 K 真实模型矩阵仍需显式执行及远端接入。这不是远端版本的验收结果。

最新远端同一 SHA 的七个已触发 workflow 均为 success：

| 工作流 | 已核实的 run |
| --- | --- |
| CI | [44249650777](https://github.com/aiqubits/quantinfer/actions/runs/44249650777) |
| Model Conformance | [44249650869](https://github.com/aiqubits/quantinfer/actions/runs/44249650869) |
| GGUF Interoperability | [44249650890](https://github.com/aiqubits/quantinfer/actions/runs/44249650890) |
| Performance Regression | [44249650800](https://github.com/aiqubits/quantinfer/actions/runs/44249650800) |
| CUDA Conformance | [44249650852](https://github.com/aiqubits/quantinfer/actions/runs/44249650852) |
| GGML Conformance | [44249650776](https://github.com/aiqubits/quantinfer/actions/runs/44249650776) |
| WGPU Conformance | [44249650714](https://github.com/aiqubits/quantinfer/actions/runs/44249650714) |

本次读取实际 job 日志，确认：

- [Ubuntu latest 三语言 job](https://github.com/aiqubits/quantinfer/actions/runs/44249650777/job/134616901658)
  与 [Ubuntu 24.04 job](https://github.com/aiqubits/quantinfer/actions/runs/44249650777/job/134616901682)
  各记录 `147 cases passed`，不只是 workflow 标题显示绿色。
- CI 资源 job 记录六个 64 MiB 组合分别经 native/runtime 在 48 MiB 地址空间下完成
  校验，共 12 个独立进程检查。此限额不等于完整模型加载或最大合法压缩块的内存上限。
- [SmolLM2 job](https://github.com/aiqubits/quantinfer/actions/runs/44249650869/job/134616902647)
  执行 F32/Q8_0/Q4_0、混合包、imatrix、原生 QIF 六组合、HTTP、C/Python、WASM/WASI-NN
  等已列步骤。Q4_0 sampled-logit max error 为 `3.11140704`，Top-10 重合 `7`，符合该
  已有 GGML layout profile 的门槛；不可直接当作另一 layout 的 K-quant 阈值。
- [Qwen3 job](https://github.com/aiqubits/quantinfer/actions/runs/44249650869/job/134617548856)
  执行独立 Transformers 参考、F32/Q8_0/Q4_0/Q4_K。Q4_K logits max `11.687058`、
  MSE `7.92543464`、Top-10 重合 `7`，通过已有 `13 / 9 / 6` 门槛。没有 Q5_K/Q6_K 的
  完整独立真实模型质量验收，不能用 SmolLM2 混合包中出现这些 dtype 代替。
- 两模型六个基准变体的同 runner 固定参考比较实际执行；即使历史静态 CPU 基线
  不可用，也有配对检查。比率用于回归门禁，不构成统计显著提速证明。
- 自定义 external-artifact job 因未提供外部输入而 skipped；三个默认 ignored 测试
  的相关固定模型/官方模板路径由模型 workflow 显式执行。没有把 skipped/ignored
  随意计为“通过”，也没有把它们全部解释为“从未验证”。

上述厂商 workflow 的成功应按真实步骤解释：编译、ABI double、软件适配器或数值
契约测试不能扩大成真实 CUDA/Metal/NPU 设备吞吐、功耗或完整模型质量的测量。

## 文档一致性与下一步判定

两份方案的阶段顺序不构成根本矛盾。通用方案要求先稳定 Manifest/GGUF 再实现 QIF；
用户后续明确采用新版 QIF 并废弃旧版，当前默认发布 QIF、继续接受 GGUF 符合这一决策。
QIF v1 私有历史代码和测试仍存在，但默认 runtime 不接受 v1，也不作为公开能力暴露。

旧的 `two-plan-acceptance-review.md` 明确固定在较早源码提交，其“默认 QIF loader
未接入”“独立语言 reader 只覆盖固定 F32”等结论不能当作当前事实。本报告不把
历史快照与新代码的差异错误地列为当前实现自相矛盾。

对方案中的论文链接也做了原文核对：
[Universal Checkpointing](https://arxiv.org/abs/2406.18820)、
[fastsafetensors](https://arxiv.org/abs/2505.23072)、
[DataStates-LLM](https://arxiv.org/abs/2601.16956)、
[TierCheck](https://arxiv.org/abs/2605.17821)、
[LayerCheck](https://arxiv.org/abs/2609.27193)、
[增量快照无损压缩论文](https://arxiv.org/abs/2505.09810) 与
[ByteCheckpoint](https://www.usenix.org/system/files/nsdi25-wan-borui.pdf)
的链接均对应所列研究。训练 checkpoint 研究提供设计启发，不是 QIF 推理性能的实证。

当前可以接受的判断是：**现有固定 CPU MVP 与 QIF core 测试矩阵通过；两份方案的
完整功能验收未通过。** 下一步应先完成 K-quant 的产物精度闭环与已有本地补丁的
真实模型回归，澄清上述 QIF 规范文字，再分别补齐高级量化算法、复杂 mixed precision
和 Studio 的端到端任务管理。不能通过扩大绿色 CI 的含义或将剩余功能改写为完成来验收。

[qif-plan]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/docs/formats/qif-v2-final-plan.md
[qif-old-state]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/docs/formats/qif-v2-final-plan.md#L48-L60
[qif-acceptance]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/docs/formats/qif-v2-final-plan.md#L520-L533
[q4k]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-quant/src/schemes/q4_k.rs#L21-L41
[q5k]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-quant/src/schemes/q5_k.rs#L21-L41
[q6k]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-quant/src/schemes/q6_k.rs#L20-L35
[gguf-writer]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/gguf/writer.rs#L183-L261
[quant-provider]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-quant/src/providers/mod.rs#L90-L156
[model-source]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer/src/cpu/model.rs#L242-L291
[backing]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/qif_v2/backing.rs
[qif-api]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/lib.rs#L17-L34
[package-writer]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/qif_v2/package.rs#L229-L279
[descriptor-ids]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/qif_v2/writer.rs#L704-L731
[descriptor-validation]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/qif_v2/reader.rs#L1219-L1233
[model-manifests]: https://github.com/aiqubits/quantinfer/tree/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/model-zoo/manifests
[model-tests]: https://github.com/aiqubits/quantinfer/tree/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-runtime/tests
[tokenizer-tests]: https://github.com/aiqubits/quantinfer/tree/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-tokenizer/tests
[model-workflow]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/.github/workflows/model-conformance.yml
[semantic]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-model/src/semantic.rs
[descriptor]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-core/src/quantized/descriptor.rs
[manifest]: https://github.com/aiqubits/quantinfer/tree/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/manifest
[runtime-provider]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-runtime/src/provider.rs
[kernel]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-kernel/src/lib.rs
[runtime]: https://github.com/aiqubits/quantinfer/tree/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-runtime/src
[ggml]: https://github.com/aiqubits/quantinfer/tree/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/adapters/ggml
[benches]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/benches/README.md
[shards]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/safetensors/shards.rs
[interop]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/.github/workflows/gguf-interop.yml
[cache]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-runtime/src/cache.rs
[profile]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-runtime/src/profile.rs
[simd]: https://github.com/aiqubits/quantinfer/tree/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/backends/cpu-simd
[adapters]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/adapters/README.md
[scheduler]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-runtime/src/scheduler.rs
[server]: https://github.com/aiqubits/quantinfer/tree/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/apps/quantinfer-server
[studio]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/apps/quantinfer-server/src/routes/studio.rs
[reshard]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/qif_v2/reshard.rs
[signature]: https://github.com/aiqubits/quantinfer/blob/e1a1c96dbc2f25bcafb9200fe49d511b27c435ae/crates/quantinfer-format/src/qif_v2/signature.rs
