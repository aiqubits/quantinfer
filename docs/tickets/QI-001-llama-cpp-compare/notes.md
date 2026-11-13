# QI-001 注意事项

- 报告放在 `benches/llama-cpp/`。不要放进 `benches/baselines/`。回归脚本会扫那个目录里的全部 JSON，这份报告的字段对不上。
- 8 线程在 M4 Pro 上会用到能效核。llama.cpp 自己也是 4 线程比 8 线程快。后面的加速比用 1、2、4。
- 单线程的绝对 tok/s 会随机器冷热变。同一次运行里的线程比值比跨会话的绝对数字可靠。
- 热点验收的后半段没有做。没有 matmul / attention / sampling 的时间占比。已知的第一原因是单行 matmul 没有并行，处理在 QI-003。
- Qwen3-0.6B 没有下到本地。数字是 SmolLM2-135M Q8_0，不是票面上的 Qwen3。
- 代码和 QI-003 在同一条 `feature/cpu-decode` 上。原先它们混在主工作区的未提交改动里。
