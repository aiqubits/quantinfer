# QI-001 里程碑

- [x] `quantinfer-cli generate --threads` 接到 CPU 后端，默认 1。
- [x] `benches/llama-cpp-compare.py` 能出一份 JSON 报告。
- [x] 工作流按钉死的 llama.cpp 构建对照，并强制 CPU（`-ngl 0`）。
- [x] Apple M4 Pro 上用 SmolLM2-135M Q8_0 跑完 1/2/4/8，列切分之前的记录已入库。
- [x] `docs/backends/README.md` 写上同机差距和下一刀是 attention。
- [ ] 用 profiler 把单步拆成 matmul、attention、sampling、其它。结论来自代码路径，不是时间占比。
- [ ] 报告进入 `benches/baselines/` 并由 `check-regressions.py` 消费。两套 JSON 结构不同，放进去会被误扫。
- [ ] 同机跑 Qwen3-0.6B。这次用的是本地已有的 SmolLM2 夹具。
