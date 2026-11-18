# 剩余实施与验收记录

目标：继续实现或解决两份方案及 [复核报告](two-plan-acceptance-review.md) 中所有剩余项。
本记录区分实际实现、自动化证据与仍待验证的范围。历史复核报告保持其固定源码基线。
此前排除厂商硬件实测的范围继续保留；具体厂商内核实现不因此免除。

| 项目 | 要求和完成证据 | 当前状态 |
| --- | --- | --- |
| QIF 默认运行时接入 | SINGLE/PACKAGE 只读控制区打开；按 tensor 加载；保留 alias、原生 metadata、完整性及预算检查；CLI/server/C/WASI-NN 真实模型生成对照 | `be72240` 本地回归及远端 SmolLM2 六种组合、HTTP/C/Python/WASI-NN 验收通过，见下方证据 |
| QIF 发布与检查工具 | SafeTensors/GGUF 可发布 SINGLE/PACKAGE 与三种 codec；原生 metadata inspect、完整 verify；参数与破损输入回归 | 发布选项及六种组合回归已实现；原生 inspect/verify 已有本地回归，等待远端验收 |
| 跨语言 reader | 独立 C/Python 扩展到已冻结编码、codec、alias、PACKAGE 和签名语义；共用正反 corpus 比较结果 | 未完成 |
| 持续 fuzz | 覆盖 package JSON/path/shard，持续运行与 corpus/失败 artifact 管理；保留快速 smoke | 未完成 |
| 必需能力 registry | 明确 feature/backend 名称、能力匹配与未知要求拒绝；无需调用者猜测未登记字符串 | 未完成 |
| 高级量化 provider | GPTQ/AWQ/AutoRound/LLM Compressor 的浮点权重与校准输入到量化产物闭环，独立参考与误差门禁；不以 import/dequantize 冒充算法 | 未完成 |
| Mixed precision 与精度矩阵 | 复杂策略与误差实验、两模型更多 K-quant 组合的真实质量验收，记录覆盖边界 | 未完成 |
| 厂商 kernel | Ascend matmul、T-MAC、KleidiAI 的实际库绑定及 executor/kernel 实现、能力声明、可执行 ABI/契约测试；硬件实测单列 | 未完成 |
| Studio | 量化任务、校准 workflow、registry/profile/package 管理与 benchmark 可视化；端到端任务、失败、取消及产物验证 | 未完成 |
| 完成审计 | 对照全部要求逐项检查代码、测试、真实模型及最新远端 CI，不以本表的状态文字代替证据 | 未完成 |

QIF 方案明确延后的 FP8/NF4、BLAKE3、远程 transport、PACKAGE 签名不混作当前 wire core 的缺陷；若后续扩展这些 profile，仍需单独冻结版本/编码与验收。

## 已完成迭代的证据

- `be72240626a05acaeb4c740003bc63ea1519c24a`：原生文件后端、两种 profile 与三种
  codec 发布路径；[CI](https://github.com/aiqubits/quantinfer/actions/runs/44225309125)
  通过。本地 201 项运行时测试通过、3 ignored；format release 87 项通过，另有
  mmap 编译失败 doctest。六个 64 MiB 样本分别经原生 reader 和默认运行时 reader
  在 48 MiB 地址空间下完成流式校验，共 12 次进程级检查。
- 同一提交的 [SmolLM2 job](https://github.com/aiqubits/quantinfer/actions/runs/44225308990/job/134536924695)
  已通过 Q4_0 的六种原生 QIF 组合（tensor/forward/generation 对照）、F32 原生
  PACKAGE 的 HTTP、C/Python binding 和 WASI-NN guest 检查；不扩大为全部 dtype
  在全部 binding 上的组合穷举，也不代表 out-of-core Dense 执行。
- 原生 `inspect`/`verify` 的本地回归保留 Map、整数宽度、逻辑 shape、alias、codec，
  区分控制区检查、全量 payload 校验与发布者信任。SafeTensors 发布另有六种组合
  对照，并修复省略可选 `architectures` 时空 GGUF array 导致发布失败的问题。
  此批本地 workspace 为 204 passed / 0 failed / 3 ignored；远端证据待提交验收。
