# QI-003 里程碑

- [x] `map_output_columns` 和行主序、output-row 两条调用。
- [x] 单线程与多线程的逐位测试。
- [x] release 下的重复 matmul 缩放测试，门槛是 1.5 倍，避免 CI 抖动。
- [x] M4 Pro 上 SmolLM2-135M Q8_0 的 4 线程 decode 达到单线程的 2.03 倍。
- [x] 同一次协议下 1 线程和 4 线程的生成 token 相同。
- [x] 曲线写进 `benches/llama-cpp/apple-m4-pro-smollm2-q8_0-column-split.json` 和 `benches/README.md`。
- [ ] 8 线程曲线。M4 Pro 的 8 线程会混进能效核，llama.cpp 自己在 8 线程也变慢，所以验收看 4 对 1。
- [ ] `check-regressions.py` 消费这份曲线。报告不在 `benches/baselines/` 的格式里。
