# QI-001 验证

记录文件：`benches/llama-cpp/apple-m4-pro-smollm2-q8_0.json`。

机器是 Apple M4 Pro。模型是 `smollm2-q8_0.gguf`，SHA-256 前缀
`197cc26187ff482f`。llama.cpp 是 `0ec4027a2`。prompt 5 个 token，生成 8 个，
上下文 64，release 构建。这一轮是列切分之前的 CLI。

| 线程 | QuantInfer decode tok/s | llama.cpp decode tok/s | 比值 | QuantInfer prefill ms |
| --- | --- | --- | --- | --- |
| 1 | 13.185 | 348.649 | 26.44 | 383.540 |
| 2 | 13.225 | 502.838 | 38.02 | 375.522 |
| 4 | 13.245 | 542.707 | 40.97 | 375.874 |
| 8 | 13.164 | 513.594 | 39.02 | 372.293 |

QuantInfer 的四档 decode 是一条平线。llama.cpp 从 1 到 4 线程上升，8 线程回落。
峰值 RSS 在 macOS 上是 `unmeasured`。报告里的说明是：llama.cpp 的 RSS 没测，
QuantInfer 的 RSS 只在 Linux 上读 `/proc`。

第一次对照作废。当时没加 `-ngl 0`，llama-bench 走了 Metal，线程越多越慢。
作废的数字没有写入这份 JSON。

列切分之后、带分项计时的 release 运行记在
`benches/llama-cpp/apple-m4-pro-smollm2-q8_0-phases.json`。`check-llama-cpp-report.py`
对这份文件返回 ok。四档线程生成的 token 都是
`6395, 1526, 30, 198, 198, 504, 808, 282`。

| 线程 | decode tok/s | matmul | attention | sampling | other | QuantInfer RSS KiB | llama.cpp RSS KiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20.026 | 394.2 ms，98.9% | 1.8 ms，0.4% | 0.84 ms | 1.7 ms | 578288 | 396640 |
| 2 | 29.799 | 264.6 ms，98.4% | 1.7 ms，0.6% | 0.82 ms | 1.8 ms | 536496 | 390560 |
| 4 | 39.586 | 197.2 ms，97.8% | 1.7 ms，0.8% | 0.79 ms | 2.1 ms | 585728 | 393376 |
| 8 | 37.720 | 207.9 ms，97.8% | 1.6 ms，0.8% | 0.83 ms | 2.1 ms | 585744 | 398224 |

百分比是这四项之和。计时包在 matmul 和 attention 调用外面，只在 `--profile` 打开时生效。
8 个 token、上下文从 6 涨到 14，attention 还没有成为热点。和 llama.cpp 的差距在 matmul 里面。
