# 验证

机器是 Apple M4 Pro。llama.cpp 是 `0ec4027a2`。SmolLM2 Q8_0 的 SHA-256 前缀是 `197cc26187ff482f`。prompt 5 个 token，生成 8 个，上下文 64，release。macOS 上没加 `-ngl 0` 的那次走了 Metal，已作废，没有入库。

## 列切分之前

`benches/llama-cpp/apple-m4-pro-smollm2-q8_0.json`

| 线程 | QuantInfer tok/s | llama.cpp tok/s | 比值 | prefill ms |
| --- | --- | --- | --- | --- |
| 1 | 13.185 | 348.649 | 26.44 | 383.540 |
| 2 | 13.225 | 502.838 | 38.02 | 375.522 |
| 4 | 13.245 | 542.707 | 40.97 | 375.874 |
| 8 | 13.164 | 513.594 | 39.02 | 372.293 |

这份文件没有 RSS，也没有分项时间。

## 列切分之后，无分项计时

`benches/llama-cpp/apple-m4-pro-smollm2-q8_0-column-split.json`

| 线程 | QuantInfer tok/s | llama.cpp tok/s | 比值 | prefill ms |
| --- | --- | --- | --- | --- |
| 1 | 19.511 | 424.741 | 21.77 | 257.409 |
| 2 | 28.432 | 532.600 | 18.73 | 149.151 |
| 4 | 39.527 | 609.328 | 15.42 | 87.885 |

4 对 1 是 2.03 倍。当次终端输出里 1 线程和 4 线程的 token 都是 `6395, 1526, 30, 198, 198, 504, 808, 282`。JSON 只存了个数 8。

## 分项计时和 RSS

`benches/llama-cpp/apple-m4-pro-smollm2-q8_0-phases.json`。`check-llama-cpp-report.py` 返回 ok。四档 token 都是上面那一串。

| 线程 | tok/s | matmul | attention | sampling | other | QuantInfer KiB | llama.cpp KiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20.026 | 394.2 ms，98.9% | 1.8 ms，0.4% | 0.84 ms | 1.7 ms | 578288 | 396640 |
| 2 | 29.799 | 264.6 ms，98.4% | 1.7 ms，0.6% | 0.82 ms | 1.8 ms | 536496 | 390560 |
| 4 | 39.586 | 197.2 ms，97.8% | 1.7 ms，0.8% | 0.79 ms | 2.1 ms | 585728 | 393376 |
| 8 | 37.720 | 207.9 ms，97.8% | 1.6 ms，0.8% | 0.83 ms | 2.1 ms | 585744 | 398224 |

百分比是这四项之和。4 对 1 是 1.98 倍。8 线程的 QuantInfer 低于 4 线程，llama.cpp 仍在上升。

## 测试

在 `feature/cpu-decode` 上通过：

```sh
cargo test --workspace --locked
cargo run --locked -p quantinfer-xtask -- bench
cargo test --release --locked -p quantinfer-backend-cpu --lib column_split
```

`xtask bench` 的退出码是 0。它在 macOS 上把 `rss_kb_before` 和 `rss_kb_after` 打成 0，因为那条路径只读 Linux 的 `/proc`。

视图单测覆盖指针相等的 reshape、写时复制、行切片、转置、narrow 和越界。映射单测覆盖 GGUF 截断、QIF 截断和 QIF 头校验失败。

debug CLI 加载 `/tmp/qi-bench/smollm2-q4_0.gguf` 得到 `generated_tokens=[7676, 271, 260, 198, 1195, 390, 4269, 284]`。这是映射路径能跑通，不是和映射前构建的逐 token 对照，也不是加载 RSS。
