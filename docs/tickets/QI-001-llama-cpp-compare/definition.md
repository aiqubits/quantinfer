# QI-001 性能真相：llama.cpp 同机对照

## 目标

在同一台机器、同一模型、同一 prompt 和同一线程数下，对照 llama.cpp 记录
prefill、decode 和 RSS，并指出 decode 慢在哪里。后面的性能票以这份记录做决定。

## 约束

- 对照双方都钉版本。模型文件用 model-zoo manifest 里的 SHA-256。
- 线程数扫 1、2、4、8，不单报一档。
- 不改内核算法。线程开关可以加，切分策略留给 QI-003。

## 上下文

仓库自有基线里，Qwen3-0.6B Q4_0 在 EPYC 上单线程 decode 大约 1.84 tok/s。
这个数字没有和同机的 llama.cpp 比过，看不出差距是两倍还是二十倍。

## 关联

无前置。QI-003、QI-006、QI-008、QI-012 用它做决策依据。

## 非目标

- 不做优化本身。
- 不换一套新的 benchmark 框架。
- 不测 GPU 后端。

## 验收

- 同机、同模型、四档线程的 decode tok/s、prefill 和峰值 RSS。
- decode 单步按 matmul、attention、sampling、其它拆开的热点清单。
- 报告能被 `benches/check-regressions.py` 消费。
- `docs/backends/README.md` 用一句话写出差距倍数和第一热点。
