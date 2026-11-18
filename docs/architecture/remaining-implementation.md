# 剩余实施与验收记录

目标：继续实现或解决两份方案及 [复核报告](two-plan-acceptance-review.md) 中所有剩余项。
本记录区分实际实现、自动化证据与仍待验证的范围。历史复核报告保持其固定源码基线。
此前排除厂商硬件实测的范围继续保留；具体厂商内核实现不因此免除。

| 项目 | 要求和完成证据 | 当前状态 |
| --- | --- | --- |
| QIF 默认运行时接入 | SINGLE/PACKAGE 只读控制区打开；按 tensor 加载；保留 alias、原生 metadata、完整性及预算检查；CLI/server/C/WASI-NN 真实模型生成对照 | `be72240` 本地回归及远端 SmolLM2 六种组合、HTTP/C/Python/WASI-NN 验收通过，见下方证据 |
| QIF 发布与检查工具 | SafeTensors/GGUF 可发布 SINGLE/PACKAGE 与三种 codec；原生 metadata inspect、完整 verify；参数与破损输入回归 | `184e7ca` 本地回归和远端 SmolLM2 六种组合的 inspect/verify/generate 已通过 |
| 跨语言 reader | 独立 C/Python 扩展到已冻结编码、codec、alias、PACKAGE 和签名语义；共用正反 corpus 比较结果 | `f651ed2` 的双环境专项及主 CI 已通过，各 147 项；报告工具预算和测试边界见 conformance 文档 |
| 持续 fuzz | 覆盖 package JSON/path/shard，持续运行与 corpus/失败 artifact 管理；保留快速 smoke | `6421b63` 三个 ASan target 的远端时间预算与 smoke 检查通过；每日定时/缓存/日志/失败 artifact 已接入；有限测试的范围见下方证据 |
| 必需能力 registry | 明确 feature/backend 名称、能力匹配与未知要求拒绝；无需调用者猜测未登记字符串 | `5c4c262` 本地与远端 CI/Model Conformance 通过；已登记四个 QIF feature、七个 backend identity；CPU/GPU/direct loader 核对实际后端；语义见 [registry](requirements-registry.md) |
| 高级量化 provider | GPTQ/AWQ/AutoRound/LLM Compressor 的浮点权重与校准输入到量化产物闭环，独立参考与误差门禁；不以 import/dequantize 冒充算法 | 未完成 |
| Mixed precision 与精度矩阵 | 复杂策略与误差实验、两模型更多 K-quant 组合的真实质量验收，记录覆盖边界 | `c8c558c` 的 K 生产端与双模型矩阵远端通过；全局校准预算策略已本地闭环并接入 CI，当前批远端验证待完成，见下方 |
| 厂商 kernel | Ascend matmul、T-MAC、KleidiAI 的实际 executor/kernel 实现、所用 SDK 绑定、能力声明、可执行 ABI/契约测试；硬件实测单列 | Ascend `8928c63`、KleidiAI `26bfa4f`、T-MAC `175873e` 专项 CI 已通过；各实现 profile 与上游编译栈/硬件测量的边界见 README |
| Studio | 量化任务、校准 workflow、registry/profile/package 管理与 benchmark 可视化；端到端任务、失败、取消及产物验证 | 未完成 |
| 完成审计 | 对照全部要求逐项检查代码、测试、真实模型及最新远端 CI，不以本表的状态文字代替证据 | 未完成 |

QIF 方案明确延后的 FP8/NF4、BLAKE3、远程 transport、PACKAGE 签名不混作当前 wire core 的缺陷；若后续扩展这些 profile，仍需单独冻结版本/编码与验收。

## 已完成迭代的证据

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
