# 剩余实施与验收记录

目标：继续实现或解决两份方案及 [复核报告](two-plan-acceptance-review.md) 中所有剩余项。
本记录区分实际实现、自动化证据与仍待验证的范围。历史复核报告保持其固定源码基线。
此前排除厂商硬件实测的范围继续保留；具体厂商内核实现不因此免除。

| 项目 | 要求和完成证据 | 当前状态 |
| --- | --- | --- |
| QIF 默认运行时接入 | SINGLE/PACKAGE 只读控制区打开；按 tensor 加载；保留 alias、原生 metadata、完整性及预算检查；CLI/server/C/WASI-NN 真实模型生成对照 | `be72240` 本地回归及远端 SmolLM2 六种组合、HTTP/C/Python/WASI-NN 验收通过，见下方证据 |
| QIF 发布与检查工具 | SafeTensors/GGUF 可发布 SINGLE/PACKAGE 与三种 codec；原生 metadata inspect、完整 verify；参数与破损输入回归 | `184e7ca` 本地回归和远端 SmolLM2 六种组合的 inspect/verify/generate 已通过 |
| 跨语言 reader | 独立 C/Python 扩展到已冻结编码、codec、alias、PACKAGE 和签名语义；共用正反 corpus 比较结果 | `dcea320` 与 `02f7d9a` 的双环境专项及主 CI 已通过，各 153 项；报告工具预算和测试边界见 conformance 文档 |
| 持续 fuzz | 覆盖 package JSON/path/shard，持续运行与 corpus/失败 artifact 管理；保留快速 smoke | `6421b63` 三个 ASan target 的远端时间预算与 smoke 检查通过；每日定时/缓存/日志/失败 artifact 已接入；有限测试的范围见下方证据 |
| 必需能力 registry | 明确 feature/backend 名称、能力匹配与未知要求拒绝；无需调用者猜测未登记字符串 | `5c4c262` 本地与远端 CI/Model Conformance 通过；已登记四个 QIF feature、七个 backend identity；CPU/GPU/direct loader 核对实际后端；语义见 [registry](requirements-registry.md) |
| 高级量化 provider | GPTQ/AWQ/AutoRound 的浮点权重与校准输入到量化产物闭环，独立参考与误差门禁；LLM Compressor 按原方案核对 Adapter | GPTQ 完整 Hessian 与整模型流程已通过远端验收；AWQ 原生缩放、clipping 和 QIF 发布已通过 `34d7756` 的[整模型专项](https://github.com/aiqubits/quantinfer/actions/runs/44532255162/job/135588360329)。AutoRound 原生 SignRound 的独立 Torch 对照、小模型和真实 SmolLM2 整模型门禁本地通过，远端证据按自己的提交记录。LLM Compressor 已有 WNA16 Adapter |
| Mixed precision 与精度矩阵 | 复杂策略与误差实验、两模型更多 K-quant 组合的真实质量验收，记录覆盖边界 | `c8c558c` 的 K 生产端与双模型矩阵、`04900bf` 的全局校准预算策略及 `1c748d9` 重跑均通过远端 Model Conformance，见下方 |
| 厂商 kernel | Ascend matmul、T-MAC、KleidiAI 的实际 executor/kernel 实现、所用 SDK 绑定、能力声明、可执行 ABI/契约测试；硬件实测单列 | Ascend `8928c63`、KleidiAI `26bfa4f`、T-MAC `175873e` 专项 CI 已通过；各实现 profile 与上游编译栈/硬件测量的边界见 README |
| Studio | 量化任务、校准 workflow、registry/profile/package 管理、LoRA 和 benchmark 可视化 | 固定 SmolLM2 已在本地通过 HTTP 校准、AWQ 发布、重分片与校验，Profile 和 LoRA 生效、失败清理、运行取消、benchmark JSON 和重启恢复。已接入 Model Conformance |
| Candle QIF 布局 | 不改变原生 output-row 的逻辑坐标；所有 packed encoding、layout、维度 profile、codec 与 alias 对照 | `00049bd` 新增显式 `PackedQifTensor` 坐标视图，保留原始 blocks；五项专项、隔离 workspace 标准检查、可选 feature Clippy 与两个 xtask smoke 通过；远端 CI 已成功 |
| 模型参数与 QIF 对齐 | 已知模型参数按语义校验，不随存储维度模式跳过；SINGLE/PACKAGE 按显式偏移支持较大物理对齐，仍拒绝重叠、倒序和未对齐块 | `df4eb15` 本地 240 次模型加载、九种 codec/alignment 和三语言 189 项都通过；七个实际触发的远端 workflow 通过，含 [Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44523217346) |
| 滑动窗口、分片资源与发布 | 两种 Dense adapter 明确拒绝不支持的滑动窗口；PACKAGE reader 以单分片句柄读取；writer 完成全量校验后发布 | 本地 288 次模型加载、64 分片与 32 描述符限制、暂存目录校验失败及清理回归、三语言 189 项通过。性能 CI 启用 `pipefail` 和补齐 CPU SIMD 变更触发；对应远端 CI 单列 |
| 分片 SafeTensors 默认输出 | 分片源逐 tensor 读取，默认输出对 SINGLE/PACKAGE 与三种 codec 逐块写入 | SmolLM2 两分片的六种 QIF 组合在本地 384 MiB 地址空间限制下完成发布和校验；所有文件 hash 与原转换流程一致。已接入 Model Conformance |
| 完成审计 | 对照全部要求逐项检查代码、测试、真实模型及最新远端 CI，不以本表的状态文字代替证据 | 未完成 |

QIF 方案明确延后的 FP8/NF4、BLAKE3、远程 transport、PACKAGE 签名不混作当前 wire core 的缺陷；若后续扩展这些 profile，仍需单独冻结版本/编码与验收。

## 已完成迭代的证据

- AutoRound 对每个线性投影使用实际输入，联合优化舍入偏移和对称裁剪，前向与最终存储均
  使用 F16 网格。独立 Torch autograd 对照包含 4/8 bit、32/64 分组与较短尾组。整模型
  按当前量化后的层传播校准输入，输出可直接运行的 Q4_0/Q8_0。SmolLM2 的固定 4 bit
  任务以 13 条校准提示、32 步和 4 线程优化 210 个投影，四条验证提示 KL 为
  0.239773、0.055974、0.024860、0.067210，Top-10 重合为 8、8、9、9。SINGLE/none、
  PACKAGE/zstd 的权重及推理对照通过。默认迭代参数和已测 profile 见
  [AutoRound](../../adapters/autoround/README.md)。

- AWQ 在 Qwen3/Llama 的真实 attention 和 SwiGLU 运算上选择缩放，逐层传递已量化输出，
  对 V/O/Gate/Up/Down 执行基于 token 重建误差的 clipping。独立 Torch 向量覆盖 4/8-bit、
  32/64 分组和尾组。48 种发布组合覆盖两架构、两维度顺序、两位宽、两容器与三 codec。
  额外验证了半精度归一化参数转为 F32、保留其它载荷、类型化元数据和失败时不发布。
- 真实 SmolLM2 校准使用 13 条输入和 159 个 token，量化全部 210 个投影并补偿 60 个归一化
  参数。四条留出输入的 KL 分别为 0.130847、0.103983、0.027359、0.163912，
  Top-10 重合均为 8，达到预设 KL ≤ 0.5、Top-10 ≥ 6。SINGLE/PACKAGE 的逐张量哈希、
  logits、生成和 KV 一致。完整本地检查 291.09 秒。模型 CI 已加入该验收作业。

- `e6a3f2f` 的实际十一项 workflow 已全部 completed/success，包括
  [完整 Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44515061181)。
  Qwen3 扩展 K 矩阵、同机性能、外部与校准 GGUF 互操作均通过。
- `e6a3f2f` 的 [GPTQ whole-model job](https://github.com/aiqubits/quantinfer/actions/runs/44515061181/job/135528822242)
  已成功，正式测试 1627.26 秒，量化／发布子进程 1613.55 秒。全部 210 个投影、
  四个固定输入的既定质量门槛和两种 QIF profile parity 均通过，四条 KL 与此前
  本地结果一致。其它模型 job 的状态仍按完整 workflow 核对。
- 固定 `e6a3f2f` 的新审查复现两项问题：原生维度 profile 跳过已知 RoPE／Value
  参数检查；规范允许的大对齐多 chunk 被三个实现一致拒绝。修复将参数校验与
  storage dimension order 分离；各 reader 按显式偏移接受额外 padding，保持
  4 KiB 基本对齐、范围、顺序和重叠检查。默认 writer bytes 不变，无新 wire ID。
  新模型矩阵包含两架构、两 profile、三种来源和两个入口：48 次合法前向值对照，
  192 次错误声明拒绝。大对齐覆盖三种 codec、4 KiB／64 KiB／2 MiB、两种容器，
  memory/file/mmap/default、alias 和跨 chunk 范围；独立三语言矩阵为 177 个
  正反用例加 12 corpus seed，共 189 项。本地通过不代替本批自身远端 CI。

- `00049bd` 的 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44510167004)
  已成功，实际执行新 Candle 矩阵、默认检查、双环境 153 项跨语言对照、资源检查、
  三个 fuzz smoke 和原生 x86/ARM SIMD；该批仅触发此一个工作流。
- 新全模型 GPTQ CLI 使用源模型、真实校准激活和前面已更新的投影，按层依次生产
  210 个 SmolLM2 Q4_0 投影。13 条声明校准输入共 159 token rows，四条验证输入
  KL 为 `0.429150 / 0.340109 / 0.092968 / 0.233445`，Top-10 为 `7 / 6 / 7 / 7`，
  均通过预先设置且未放宽的 `0.5 / 6` 门槛；首轮 CLI 耗时 979.69 秒。
  此前三提示 21-row 的 KL `0.552594` 失败未被改写为成功，也未发布其产物。
  提示来源及开发/验证边界见 [fixtures](../../tests/fixtures/gptq/README.md)。
  新正式验收测试随后实际重算全部 210 个投影，并检查两种 QIF profile 的全部
  raw hash、完整 logits、确定性 generation 和每层 KV，已通过；算法/发布子进程
  耗时 1012.82 秒，完整测试 1023.28 秒。四条质量结果与首轮完全一致。
  原有单投影的更严格门槛和独立八组 upstream vectors 保留；全模型固定样例
  不扩大为其它模型/参数的通用任务准确率。新独立远端 job 待对应提交确认。
  本地标准 workspace 为 257 项运行时测试通过、7 ignored，另 1 项 compile-fail
  doctest 通过；gptq feature 的五项全流程集成检查、strict Top-K 单测、可选
  Clippy 和 format description release 回归通过。F16/BF16 输入的非投影 payload
  按原精度逐字节保留；模型包来源 revision 与校验和失败路径单独验证。

- 对 `02f7d9a` 的独立审查发现 Candle packed 导入忽略原生 output-row 坐标：
  合法 Q8_0 `[32,64]` 的逻辑元素从 -10 被误读成 -12；原有两个可选 feature 测试
  仍通过。修复保留物理 packed storage 和显式 logical view，`dequantize` 应用
  transpose stride；同时验证所选 tensor raw hash，支持 none/zstd/LZ4 和 alias。
  新矩阵为五种 encoding × 两种 layout × 两种维度 profile × 三种 codec，共
  60 个 artifact，每个对照 physical/alias 的全部逻辑元素及原始 bytes。
  另保留原始非对称反例、output width=3、有限 payload 损坏拒绝与 unaligned 输入。
  后端文档中旧 contract-only/无 FFI 的描述和跨语言数量已按实际实现修正。
- `02f7d9a` 的十一项远端工作流已全部成功，包括
  [Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44501323457)。
  Qwen3 的原有独立参考、三种 K 质量/六种 QIF 对照、同机性能门禁、上游 GPTQ
  八组向量重算及 llama.cpp 互操作均通过。这些已有检查不覆盖上述新 Candle 反例。

- `be72240626a05acaeb4c740003bc63ea1519c24a`：原生文件后端、两种 profile 与三种
  codec 发布路径；[CI](https://github.com/aiqubits/quantinfer/actions/runs/44225309125)
  通过。本地 201 项运行时测试通过、3 ignored；format release 87 项通过，另有
  mmap 编译失败 doctest。六个 64 MiB 样本分别经原生 reader 和默认运行时 reader
  在 48 MiB 地址空间下完成流式校验，共 12 次进程级检查。
  [Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44225308990)
  的两模型 job、数值/生成/同机性能门禁均已通过；custom external-artifact job
  因无输入按配置 skipped。
- 同一提交的 [SmolLM2 job](https://github.com/aiqubits/quantinfer/actions/runs/44225308990/job/134536924695)
  已通过 Q4_0 的六种原生 QIF 组合（tensor/forward/generation 对照）、F32 原生
  PACKAGE 的 HTTP、C/Python binding 和 WASI-NN guest 检查；不扩大为全部 dtype
  在全部 binding 上的组合穷举，也不代表 out-of-core Dense 执行。
- 原生 `inspect`/`verify` 的本地回归保留 Map、整数宽度、逻辑 shape、alias、codec，
  区分控制区检查、全量 payload 校验与发布者信任。SafeTensors 发布另有六种组合
  对照，并修复省略可选 `architectures` 时空 GGUF array 导致发布失败的问题。
  此批 `184e7ca` 本地 workspace 为 204 passed / 0 failed / 3 ignored；
  [远端 CI](https://github.com/aiqubits/quantinfer/actions/runs/44226860535) 已通过，
  [SmolLM2 job](https://github.com/aiqubits/quantinfer/actions/runs/44226860652/job/134542024412)
  已通过原生六种组合的真实模型 inspect/verify 和生成对照。
- 三个 ASan fuzz target 的本地 60 秒时间预算运行均正常结束（libFuzzer 实际
  记录 61 秒）：reader 201,869、PACKAGE index 148,249、PACKAGE controls 233,865
  次；coverage 均非零。PACKAGE controls 的 90 个基线 seed 覆盖三种 codec、alias、
  重复/过深 JSON、metadata、shard 与路径；每日任务设置 900 秒/target。
  有限 fuzz 没有证明全部拓扑、并发文件系统或所有恶意输入安全。
- `6421b63` 已接入每日持续 fuzz 与 smoke artifact 保存。90 个确定性基线 seed
  之外还保留了首轮探索得到的 coverage corpus；[远端 CI](https://github.com/aiqubits/quantinfer/actions/runs/44227476116)
  与 [Sustained Fuzz 首轮](https://github.com/aiqubits/quantinfer/actions/runs/44227476126)
  均成功。该 push 首轮各运行 61 秒，reader 94,344、PACKAGE index 59,713、PACKAGE
  controls 86,674 次；三个 job 都保存了日志/corpus artifact。每日 900 秒配置
  不能解释为首轮已经运行 900 秒，也不以定时任务代替后续失败排查。
- 必需能力 registry 的本地 workspace 为 206 passed / 0 failed / 3 ignored，
  Clippy 通过；另验证 WGPU/CUDA/CPU SIMD 编译及 Ascend native 契约测试，
  这些编译/契约证据不是厂商设备实测。
  `5c4c262` 的 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44228340333)
  与 [Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44228340266)
  均成功；`184e7ca` 的完整 [Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44226860652)
  也已通过两模型门禁。
- Ascend native 专项复核发现两个公开方法可将无效裸指针传入 CANN 的 P1 安全
  边界问题。prepare/execute 现有明确 unsafe 契约，执行消费单次 plan；两个
  compile-fail doctest 和 native Clippy 门禁防止回退。另修复 aclInit 成功后
  device query 失败路径未 finalize 的清理遗漏。`d1af7c9` 的
  [专项工作流](https://github.com/aiqubits/quantinfer/actions/runs/44228896531)
  与 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44228896402) 均成功。
- 随后的实际 [Ascend executor](../../backends/ascend/README.md) 管理 F32 tensor、
  设备分配、拷贝、context/stream、两阶段 ACLNN、显式 executor 销毁和同步；CLI
  增加可选 `ascend`。独立 C double 延迟执行到同步，检查结果、调用次数、线程迁移、
  context 恢复和各失败阶段资源清理；未知完成状态保留资源并隔离运行时。
  本地默认 workspace 207 passed / 0 failed / 3 ignored；native 另有 5 项运行时
  测试通过、2 项编译失败 doctest，通过的契约测试不计作 NPU 测量。一个真实设备
  F32 fixture 测试明确 ignored；本轮不把未执行的硬件测试标为通过。
- `8928c63` 的 [Ascend 专项](https://github.com/aiqubits/quantinfer/actions/runs/44232157163)
  和 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44232157312) 已通过。
- KleidiAI 选用上游 `9b65dd111ab5cc8a6c0d3c9c0883b69ad97eeab5` 的真实 NEON
  dotprod kernel，精确保留 RHS Q4_0/F16 scale，并单独量化激活。五种 shape ×
  两种激活分布通过本地 QEMU 执行，另有输入范围拒绝测试；不是以 C mock 替代
  实际 kernel。默认 workspace 208 passed / 0 failed / 3 ignored，跨 AArch64
  Clippy 和 CLI 进程内单测通过。本地跨 CLI 子进程测试返回 127，未记为通过；
  原生 ARM CI 将执行相同的 CLI 集成检查。完整模型质量/能耗性能另列。
- `26bfa4f` 的 [KleidiAI 原生 ARM job](https://github.com/aiqubits/quantinfer/actions/runs/44236519435/job/134573867442)
  已执行真实 SDK，5 项 kernel/packing 检查和 7 项 CLI 检查均通过，包含本地仿真
  子进程未通过的两项集成测试；Clippy 通过。没有将 QEMU 失败改记为本地通过。
- T-MAC 的具体实现是原生 Rust LUT4 profile，包含 AVX2 gather 与 NEON TBL，
  不调用上游 TVM 编译/调优栈。65,536 种 plane mask、F64 参考矩阵、prepared
  snapshot 和错误输入在本地 x86 与 AArch64 QEMU 均通过；CI 将验证原生双架构。
  保留上游/下游 kernel hook，量化位宽和数值范围不扩大到该 profile 之外。
- `175873e` 的 [T-MAC 双架构专项](https://github.com/aiqubits/quantinfer/actions/runs/44239948395)
  已通过：原生 x86 job `134585096877` 记录 `verified T-MAC LUT4 Avx2`，
  ARM job `134585096214` 记录 `verified T-MAC LUT4 Neon`，各 5 项 kernel/packing
  与 7 项 CLI 检查通过；[CI](https://github.com/aiqubits/quantinfer/actions/runs/44239948294)
  也已通过。不将上述正确性检查改称完整模型性能测量。
- `4c880a4` 的独立 Python core-profile reader 不调用 QuantInfer binding，使用标准库解析
  JSON/QMB/QIDX、原生 Python LZ4、系统 libzstd streaming 和 OpenSSL Ed25519。
  本地 106 个正反比较通过，包含八种 encoding × 两种 layout、scalar、alias、
  codec/profile 组合、跨 chunk/shard 重建、root 修复后的语义错误及公钥信任策略。
  该批工具按 artifact/decoded physical tensor 各 64 MiB 设范围；当时原 C reader 尚未扩展，
  随后的 C 扩展及 alias 报告预算收紧见下方记录。
  本地默认 workspace 212 passed / 0 failed / 3 ignored，format all-targets Clippy 通过；
  新例程与 CI 使用相同的 fixture generator 和比较脚本。
- `4c880a4` 的 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44242382774)
  已通过，job `134593113311` 的日志明确记录 Python/native 的 106 项检查通过。
- 独立 C core-profile reader 使用 JSON-C、liblz4/libzstd 与 OpenSSL，不链接
  QuantInfer/Rust/Python。新矩阵有 135 个专门用例及 12 个 corpus seed；
  本地三语言比较通过，C 使用 ASan/UBSan。NUL 字符串和 map key 使用长度保留，
  C 的内部键映射可逆且覆盖前缀碰撞/重复键；不引入 QIF wire 字段。
  三个报告工具把 alias 计入 64 MiB 总逻辑 tensor 报告预算，并明确与格式大小限制区分。
  此轮对照还修复生产 reader/writer 共用的 manifest 字符串检查遗漏 checksum 路径的问题。
- 此轮还复现宽松 Ed25519 验证接受 identity 弱公钥／identity R／zero S 的退化
  签名；已将 SINGLE profile 明确为 canonical、拒绝 small-order A/R、canonical S
  与严格验证等式。Rust 改用 `verify_strict` 并重新编码核对公钥，独立 C/Python
  在 OpenSSL 前预检相同规则，原始有效 writer 签名不受影响。测试包含显式 pin 弱 key
  和改 root 的拒绝，另保留 valid-signed 与 weak-key-forgery fuzz seed。
  该问题不能用于冒充正常的已受信公钥；默认 `publisher_trust=not_evaluated` 的边界不变。
  本地默认 workspace 为 215 passed / 0 failed / 3 ignored，workspace all-targets
  Clippy 通过；GCC 与 Clang 两次 ASan/UBSan 三语言矩阵各 147 项通过。
  加入两项签名 seed 后，reader ASan smoke 完成 1024 runs、非零 coverage（cov 3228），
  没有把 smoke 的短时运行写成持续 fuzz 或穷举证明。
- `1f802f8` 的 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44246832175)
  暴露 Ubuntu 26.04 / JSON-C 0.18 的有效中文输入被 strict parser 拒绝，本轮未记为通过。
  已用上游 `json-c-0.18-20240915`（`41a55cf`）在本地复现；0.18 在 signed-char 构建下
  将非 ASCII 字节当作控制字符。独立 UTF-8 校验后，仅对依赖 parser 的缓冲区采用
  等价 ASCII Unicode escape；strict mode 与原始 UTF-8 canonical 字节比较保留。
  本地 0.17、0.18 均通过 147 项；CI 拆出 Ubuntu 24.04/latest 双环境，等待修复批远端结果。
- `f651ed2` 的 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44248704432)
  已全部通过。两项独立语言 job 各完成 147 个检查：
  [Ubuntu 24.04](https://github.com/aiqubits/quantinfer/actions/runs/44248704432/job/134613834702)
  实际依赖为 JSON-C 0.17 / LZ4 1.9.4 / zstd 1.5.5 / OpenSSL 3.0.13；
  [Ubuntu latest](https://github.com/aiqubits/quantinfer/actions/runs/44248704432/job/134613834392)
  为 0.18 / 1.10.0 / 1.5.7 / 3.5.5。没有用固定旧依赖绕过失败。
- 随后的量化门禁复核复现有限权重经 F32 累加得到 `Inf/NaN`、小数值经平方
  underflow 或 EPSILON 分母抬高后被低估的问题。统计与误差计算改用 F64 累加，
  relative L2 使用真实范数比；两个零 tensor 返回 0，零参考配非零误差或超出
  有限 F32 表示范围的报告明确报错，公共报告字段类型保持 F32。
  三项回归在旧实现上均失败，修复后 quant crate 12 项通过；本地 workspace
  218 passed / 0 failed / 3 ignored，workspace Clippy 与两个 xtask bench smoke 通过。
  这项诊断修复不等于已经完成复杂 Mixed Precision 或两模型扩展质量矩阵。
- 后续独立码表枚举复现 Q4_K/Q5_K/Q6_K 在量化局部 scale/min 和 F16 全局
  参数后未重新选码的 P2 问题。原生生产端现按最终存储码表选码，并处理有限 F16
  范围及正 subnormal 参数。六项专项测试包含全码点枚举、零值、极小值和范围拒绝；
  不将逐元素重建误差改善等同于所有模型的质量改善。
- 新固定模型矩阵准备 native row-major Q4_K/Q5_K/Q6_K，并对照 SINGLE/none、
  PACKAGE/zstd 的所有 tensor shape/dtype/raw hash、query 反量化值、完整 logits、
  generation/stream 和 KV。CI 按变体临时发布、保存产物 hash 后清理，避免同时
  保留六份大模型 QIF；源 checkpoint、converter 与 GGUF hash 单独记录。
  现有模型门槛不变，新矩阵的 KL/Top-K/权重预算及边界见
  [量化文档](../quantization/README.md#native-k-quant-storage-and-quality-checks)。
  本地 SmolLM2 新矩阵及临时发布路径通过，Qwen3 原有独立参考与 Q4_K 门禁、三种
  K 格式及六个 QIF 组合均通过。Qwen3 新矩阵实际耗时 1072.34 秒；CI 相应保留
  明确的 job 时间预算。六种变体的固定提示 KL 分别为 SmolLM2
  `0.095299 / 0.086190 / 0.031238`、Qwen3
  `0.031882 / 0.005511 / 0.000218` nats，均通过新增 0.5 nats 门槛。
  Qwen3 F32/Q8_0 产物 hash 与原始基线一致；已有 Q4_K raw-logit 门槛仍通过。
  本轮提交范围的 workspace 为 225 passed / 0 failed / 4 ignored，Clippy 和两个
  xtask bench smoke 通过。远端扩展矩阵仍待确认，不以本地结果代替对应 CI。
- QIF 规范中的 v1 历史加载描述、公开 reshard 名称、物理/逻辑 offset 对齐范围和
  core dtype/encoding 合法配对已明确。固定 `e1a1c96` 的新复核报告保留原基线，
  不把实现边界改写成已完成高级量化或完整 Studio。
  `62644a1` 的 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44264709769)
  六个 job 均通过。
- `c8c558c` 的八项工作流均成功，包括 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44266091791)
  与 [Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44266091876)。
  [SmolLM2 job](https://github.com/aiqubits/quantinfer/actions/runs/44266091876/job/134670856824)
  和 [Qwen3 job](https://github.com/aiqubits/quantinfer/actions/runs/44266091876/job/134671746180)
  各实际记录三种 K quality、六次 QIF parity，包含每份临时产物的 hash；数值与上述
  本地结果一致。原有独立参考、生成、RSS/KV、同机性能门禁也通过。
- 新的 `--mixed-precision-optimize` 从浮点投影的实际存储精度重建误差建立候选，
  用精确 Pareto frontier 在全局逻辑 payload 字节预算内选型，可用实际激活 imatrix
  的对角权重。solver 有独立穷举、反贪心样例、alias group、确定性、溢出、不可行
  与资源耗尽检查；达到上限明确失败，不返回假称最优的近似结果。转换器保留两个
  pass 的输入 hash 检查、实际编码成本、选择和产物 hash 报告。该预算不是容器
  大小或进程内存预算，不把校准代理目标称为任务准确率。
- 原生 block 的 `with_storage_precision` / `error_report_for_storage` 显式评估 F16
  scale 的存储精度；F32 Core 对象和旧 quantizer 的输出 bytes 保持原语义。
  三个格式 round-trip 回归覆盖六种数据变体、极小值和范围拒绝，并与实际 GGUF
  重读结果比较。新模式关闭时，Qwen3 F32/Q8_0 的完整文件 hash 仍与固定基线一致。
- 本地 SmolLM2 以三个独立声明的提示采集 210 个投影的 imatrix，在 195,000,000
  字节上限内选出 194,999,328 逻辑 payload 字节：Q4_K 6、Q5_K 91、Q6_K 85、
  Q8_0 28 个投影。固定质量提示的 KL 为 0.0084698 nats、Top-10 重合 10，
  生成 token 与 F32 golden 一致；SINGLE/none 与 PACKAGE/zstd 的所有 tensor
  inventory/raw hash、logits、generation、stream/KV 对照通过。该校准预算方案的
  真实质量样例目前是 SmolLM2，不扩大为任意模型和政策；Qwen3 的 K 矩阵单独验收。
  本地 workspace 合计 236 passed / 0 failed / 5 ignored，Clippy、两个 xtask smoke
  通过；当前批校准预算 workflow 的远端结果待对应提交验证。

- `04900bf` 的八项远端工作流均成功，包括 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44274357677)
  与 [Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44274357759)。
  SmolLM2 的校准预算优化、两个 QIF profile、固定模型数值/生成、绑定与服务检查均通过；
  Qwen3 的三种 native K 及六次 QIF 对照、原有数值门禁、三变体同机性能和 llama.cpp
  互操作也已通过。workflow 成功不覆盖随后审查新增的布局与外部适配器反例。
- 新一轮对固定 `04900bf` 的独立审查复现 QIF 逐 tensor layout 分派、GPTQ 零点
  方向/位宽回绕、外部解码 F32 溢出和 AutoRound 配置类型遗漏；原标准检查仍为
  236 passed / 0 failed / 5 ignored，加入九个审查用例后为 237 passed / 8 failed / 5 ignored。
  这些是原提交的事实，不将后续修复反写成原版本已经通过。
- 外部适配器修复批将标准 GPTQ group-major 零点表置于默认优先级；增加显式四种
  zero-point layout API，避免方形转置表的歧义。零点偏移在 code bit width 内回绕，
  三个独立 importer 拒绝重建 F32 溢出，AutoRound 校验已提供字段的类型和整数范围。
  原六个适配器反例已转为通过；另有 2/4/8-bit × 四种 offset × 四种 layout 的
  48 组独立模块算术比较、显式布局拒绝和 AutoRound 转发/配置别名测试。
  本批不替代 GPTQ/AWQ/AutoRound 的校准算法生产闭环；QIF 布局问题另行修复。

  本批本地使用独立 target 目录完成 fmt/check/workspace test/Clippy 和两个 xtask
  smoke：245 passed / 0 failed / 5 ignored。该结果不复用审查工作区的 test binary，
  对应提交的远端 CI 另行核对。

- 布局修复时更正了上一轮审查的一处反例假设：GGML 兼容 profile 的逻辑矩阵
  已是 `[N,K]`，不能在改标 row-major 后额外转置 bytes 并声称权重未变。
  采用明确坐标规则、非对称权重和独立 target 重测旧 `04900bf`：原生 QIF
  混合布局被默认 factory 拒绝，直接模型入口产生错误 logits；GGML 兼容 profile
  的等价权重没有失败。原审查中“GGML 混合布局导致 0.9860505”不作为已证实缺陷。
- 新实现按 tensor descriptor 分派原生 QIF 的 packed/output-row 线性权重，并
  规范化原生浮点 embedding/LM head 的布局；模型入口核对实际布局，CPU/GPU
  preflight 对同一 dtype 的不同布局分别匹配。GGML 兼容维度和 Llama Q/K 通道
  变换保持原约定，没有改变已有正常产物的 bytes 或 ID。
  八种核心编码 × 两种布局 × 两种维度 profile 的非方形矩阵对 F64 参考通过，
  quantized blocks 保持一致；两个 Dense adapter 的 72 组容器/codec/布局场景、
  各自默认和直接入口的 prefill/decode/KV 对照通过。
  QIF 规范补充坐标表、SHA-256 ID、QMB bool、LZ4 framing 和 QSIG 字段。
  坐标定义区分矩阵执行 profile 与低层 raw IO，保留既有 scalar/vector 退化布局；
  不把任意高 rank 的读写解释为已经支持 batched matrix kernel。独立三语言检查继续验证。

  复核坐标规则时还将已有的 reserved dimension-order bool 约束前移到原生 QMB
  writer/reader 控制区边界；C/Python 独立 reader 同步核对顶层类型，未知 hint
  内部的同名 key 仍为普通数据。新增四种错误类型的 rehashed 反例和两个 bool
  正例，避免“格式 verify 已通过、仅模型入口才发现结构类型错误”的不一致。

- `1c748d9` 的 [CI](https://github.com/aiqubits/quantinfer/actions/runs/44283564873)
  与 [Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44283564982)
  已完成并成功；两个独立跨语言 job 各执行 147 项。Qwen3 的三种 K 质量矩阵、
  六次 QIF 对照、同机性能和 llama.cpp 互操作也通过。这些结果不包含尚未提交的布局修复。
- 本地布局与结构类型修复批为 251 passed / 0 failed / 6 ignored，含 doctest；
  fmt/check、workspace all-targets Clippy、两个 xtask smoke 和五项 benchmark 工具测试通过。
  独立 Rust/C/Python ASan/UBSan 矩阵实际重跑 153 项通过。
  显式运行新增真实 SmolLM2 布局测试也通过：SINGLE/none、PACKAGE/zstd 各核对
  272 个 tensor 的 bytes/layout/shape，完整 logits 与源 F32 模型相同，固定生成 token
  一致，30 层 KV 均为 13。源 GGUF SHA-256 为
  `2549efafb6629d3b44a3e1eba18bc2a222fcca24302fc2b7abe3f8b2d17cc762`；
  仅改变两个非方形投影的物理布局，不重新量化。该批自身的远端 CI 在提交后核对。

固定 `1c748d9` 的 [全条款复核](two-plan-review-1c748d9.md) 保留远端与本地的原始状态，
不将后续修复反写成原提交已经通过。高级量化的全部生产流程与完整 Studio 仍未完成。

- `dcea320` 已发布 QIF 逐 tensor layout、结构类型校验和规范精确化修复。该提交的
  常规 CI、Metal/WGPU/CUDA、Ascend、KleidiAI、T-MAC、GGUF 互操作及 push 时间预算
  fuzz 已成功；[完整 Model Conformance](https://github.com/aiqubits/quantinfer/actions/runs/44492248035)
  也已成功。SmolLM2 新增真实布局门禁、Qwen3 三种 K 的六次 QIF 对照、同机性能
  和 llama.cpp 互操作均通过；该提交实际触发的十项 workflow 已全部成功。
- 新的 GPTQ provider 不再只有 import/dequantize：它累计完整 F64 Gram、对实际
  激活 Hessian 加阻尼并求 inverse-Cholesky，执行逐通道误差反馈，支持静态分组及
  activation order。F16 scale 在选码和反馈前冻结，4/8-bit 结果直接映射现有
  Q4_0/Q8_0 output-row blocks；无校准或预算不足明确失败，不降级为 RTN。
  固定上游 `2d65066` 的原始 Torch 实现独立生成八组码/scale 对照，Rust 全部逐项一致，
  三种 codec 的 QIF 产物及原生 CPU 对照通过；生产 Rust API 不依赖 Torch/CUDA。
- 新的真实 SmolLM2 单投影检查使用三个明确提示采集 21 个 token row：首层 query
  的 GPTQ 投影 MSE 为 `0.000013752617658749938`，同一最终码表的 RTN 为
  `0.002377514089817588`。SINGLE/none 与 PACKAGE/zstd 均保留相同量化 blocks，
  holdout KL 为 `0.0000022303983379325617`、Top-10 重合 10，生成 token 与 F32
  golden 一致。该结果不表示所有层均已 GPTQ，也不扩大为一般任务质量保证；
  全模型编排、AWQ、AutoRound 和完整 Studio 继续保留为未完成项。
  本批最终本地 workspace 为 257 passed / 0 failed / 7 ignored，含 doctest；
  fmt/check/Clippy 与两个 xtask smoke 全部通过。新增真实模型用例已单独执行成功，
  上游原始源码重算八组 fixture 也通过；远端结果按对应提交的 CI 另行核对。
- 首次取得每日 900 秒配置的实际执行证据：固定 `1c748d9` 的
  [schedule fuzz](https://github.com/aiqubits/quantinfer/actions/runs/44489991794)
  三个 job 均成功，libFuzzer 各记录 901 秒。reader 为 1,005,919 runs / cov 4092，
  PACKAGE index 为 776,352 / cov 2287，PACKAGE controls 为 1,977,629 / cov 9066；
  corpus 和日志已保存。该实际时长补齐此前记录中的证据缺口，仍不是安全穷举证明。
