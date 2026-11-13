# QI-001 至 QI-005

这五张票是同一件事：把稠密小模型的 CPU decode 测清楚，再改加载和 matmul。它们不是五张已经关闭的票。Qwen3-0.6B 的 GGUF 对照已经有了。整组还差 Q4_0 QIF 的加载 RSS，以及 SmolLM2 带分项计时的那次 4 对 1 是 1.98 倍。

代码在 `feature/cpu-decode`，从上游 `ee63330` 接出。

## 顺序

1. QI-001 同机对照 llama.cpp，先有数字再改内核。
2. QI-003 单 token decode 按输出列切分 matmul。依赖 QI-001 的根因。
3. QI-004 F32 行主序存储共享，`TensorView` 带 offset。
4. QI-005 按路径映射 GGUF 和 QIF。依赖 QI-004，并且要有 QI-001 的加载基线才能谈 RSS。
5. QI-002 把协作约定写进 `AGENTS.md`，并改掉 Runbook 里已经过时的句子。它不改运行时。本该等 QI-001 关闭后再做。

## 目标

- 同一台机器、同一模型和同一线程档上，对照钉死的 llama.cpp，记下 decode、prefill、峰值 RSS，以及 decode 时间落在 matmul、attention、sampling 还是其它。
- 单行 decode 的量化 matmul 按输出列并行。`threads = 1` 与标量路径逐位相同。
- 行主序 F32 `reshape` 共享存储。视图能表示行切片、转置和带 offset 的连续区间。越界返回错误。
- 打开 GGUF 或 QIF 路径时映射文件，不再把整文件读进第二块 `Vec`。截断和校验失败要报错。
- `AGENTS.md` 复述 developer guide。Runbook 里的多请求和在线平台表述与 `quantinfer-server` 一致。

## 约束

- 对照双方钉版本。模型文件用 manifest 里的 SHA-256。
- 标量 CPU、core、kernel、format 保持 `forbid(unsafe_code)`。映射的唯一 `unsafe` 在 `quantinfer-map`。
- 不引入 rayon，不用常驻线程池。
- 对照 JSON 不放进 `benches/baselines/`。`check-regressions.py` 只认 real-model 的 key=value 报告。

## 非目标

- 不把量化权重留在映射里直接计算。解码仍然复制成运行时张量。
- 不做 attention 融合，那是 QI-006。这次 8 token 的测量里 attention 不到 1%。
- 不跑 Qwen3-Next 或 Qwen3.8-Flash-Next。仓库的模型是稠密 Qwen3 和 Llama。
- 不改作者直接提交到 main 的政策。

## 验收

整组关闭要同时满足：

- Qwen3-0.6B 与 SmolLM2 都有同机、1/2/4/8 线程的对照，含 RSS 和分项时间。
- SmolLM2 Q8_0 的 4 线程 decode 至少是单线程的 2 倍，且各线程 token 相同。
- 视图、reshape 共享、越界错误有测试，工作区测试通过。
- 映射路径的截断和校验测试通过。加载峰值 RSS 是否下降，用 Qwen3-0.6B Q4_0 的 QIF 对照现有路径。票面写过 30%，当前实现达不到，因为权重仍会解码复制。
- `AGENTS.md` 和 Runbook 与当时的代码一致，全套本地预检在同一颗提交上通过。
