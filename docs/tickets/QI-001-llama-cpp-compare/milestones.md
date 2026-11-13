# QI-001 里程碑

- [x] `quantinfer-cli generate --threads` 接到 CPU 后端，默认 1。
- [x] `benches/llama-cpp-compare.py` 能出一份 JSON 报告。
- [x] 工作流按钉死的 llama.cpp 构建对照，并强制 CPU（`-ngl 0`）。
- [x] Apple M4 Pro 上用 SmolLM2-135M Q8_0 跑完 1/2/4/8，列切分之前的记录已入库。
- [x] `docs/backends/README.md` 写上同机差距和下一刀是 attention。
- [x] decode 按 matmul、attention、sampling、其它计时。8 token 的 release 运行里 matmul 约占 98%，attention 不到 1%。
- [x] 报告由 `benches/check-llama-cpp-report.py` 检查。不放进 `benches/baselines/`，因为 `check-regressions.py` 只认 real-model 的 key=value 报告，混进去会扫错文件。
- [ ] 同机跑 Qwen3-0.6B。钉死的权重是 1.5 GB 的 SafeTensors，本地没有。数字仍是 SmolLM2-135M Q8_0。
