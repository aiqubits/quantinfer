# QI-003 里程碑

- [x] `map_output_columns` 和行主序、output-row 两条调用。
- [x] 单线程与多线程的逐位测试。
- [x] release 下的重复 matmul 缩放测试，门槛是 1.5 倍，避免 CI 抖动。
- [x] M4 Pro 上 SmolLM2-135M Q8_0 的 4 线程 decode 达到单线程的 2.03 倍。
- [x] 同一次协议下 1、2、4、8 线程的生成 token 相同，并写进 phases 报告。
- [x] 曲线写进 `benches/llama-cpp/` 和 `benches/README.md`。8 线程也补进了 phases 报告：QuantInfer 37.720 tok/s，低于 4 线程的 39.586。
- [x] release 重复 matmul 测试通过。
- [ ] `check-regressions.py` 消费这份曲线。报告格式不同，由 `check-llama-cpp-report.py` 检查。
