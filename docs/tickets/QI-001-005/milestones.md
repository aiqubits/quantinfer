# 里程碑

按依赖顺序。未勾的项挡住整组关闭。

## QI-001 对照

- [x] `--threads`、对照脚本、钉死 llama.cpp 的工作流，以及 `-ngl 0`。
- [x] 列切分之前的 M4 Pro / SmolLM2-135M Q8_0 曲线。decode 在 1/2/4/8 线程是一条平线，约 13.2 tok/s。
- [x] 列切分之后的分项计时、双方 RSS、以及四档 token 列表。见 phases 报告。
- [x] `check-llama-cpp-report.py` 检查这份报告。报告不进 `benches/baselines/`。
- [ ] Qwen3-0.6B。钉死权重是 1.5 GB SafeTensors，本地没有。

## QI-003 列切分

- [x] 行主序与 output-row 的列切分。单线程逐位测试。
- [x] release 下 `column_split_decode_shape_scales_over_repeated_matmuls` 通过。
- [x] 早先一次 4 对 1 是 39.527 / 19.511 = 2.03 倍。
- [x] phases 报告里 1/2/4/8 的 token 相同。同一次 4 对 1 是 39.586 / 20.026 = 1.98 倍。
- [ ] 2 倍要在 Qwen3-0.6B 上再看一次。SmolLM2 的比值会随机器冷热落到 2 倍上下。

## QI-004 视图

- [x] F32 `Arc`、行切片、转置、narrow、指针相等的 reshape、写时复制、越界错误。
- [x] 提交 `6b8935a`。并入后面的改动之后，工作区测试里这些单测仍然通过。
- [ ] 量化张量的存储视图。票面把它列为非目标。

## QI-005 映射

- [x] `quantinfer-map`、路径映射、包校验、截断和 QIF 头校验测试。提交 `77db1f3`。
- [x] debug CLI 能加载 `/tmp/qi-bench/smollm2-q4_0.gguf` 并生成 8 个 token。
- [ ] 量化张量直接借用映射里的打包字节。
- [ ] Qwen3-0.6B Q4_0 的 QIF 加载时间和峰值 RSS。30% 这条线当前设计达不到。

## QI-002 规程

- [x] `AGENTS.md`、developer guide 指针、Runbook 三处句子。对照 commit `ee63330`。
- [x] 在 `feature/cpu-decode` 上重跑 `cargo test --workspace --locked` 和 `xtask bench`。
- [ ] 等 QI-001 的 Qwen3-0.6B 跑完后，再把这张票算进已验收。它不该早于对照结束。
