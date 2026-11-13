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

## Qwen3-0.6B

权重是 Hub revision `c1899de289a04d12100db370d81485cdf75e47ca` 的 `model.safetensors`，1,503,300,328 字节，SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。`config.json`、`tokenizer.json`、`tokenizer_config.json` 也对上了 manifest。GGUF 用本分支的 `quantinfer-convert --gguf-layout ggml --tokenizer-pre qwen2` 导出。Q8_0 文件 SHA-256 前缀 `3705b0095f97f56e`，196 个 Q8_0 张量和 115 个 F32。Q4_0 前缀 `f2f6fe977e658346`，196 个 Q4_0。两份报告 `check-llama-cpp-report.py` 都是 ok。

`benches/llama-cpp/apple-m4-pro-qwen3-0.6b-q8_0.json`。四档 token 都是 `3, 4, 3, 4, 3, 4, 3, 4`。

| 线程 | tok/s | llama.cpp | 比值 | matmul | attention | QuantInfer KiB | llama.cpp KiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.852 | 48.248 | 16.92 | 2788.3 ms，99.5% | 6.9 ms，0.2% | 3380304 | 2300144 |
| 2 | 5.408 | 83.852 | 15.51 | 1465.3 ms，99.0% | 6.6 ms，0.4% | 3400464 | 2304832 |
| 4 | 9.892 | 139.903 | 14.14 | 793.7 ms，98.2% | 6.4 ms，0.8% | 3400240 | 2304864 |
| 8 | 15.041 | 105.161 | 6.99 | 517.9 ms，97.4% | 6.1 ms，1.1% | 3400464 | 2300832 |

4 对 1 是 3.47 倍。8 线程时 QuantInfer 还在升，llama.cpp 从 4 线程的 139.9 掉到 105.2。

`benches/llama-cpp/apple-m4-pro-qwen3-0.6b-q4_0.json`。四档 token 都是 `3, 4, 3, 5, 3, 4, 3, 5`。

| 线程 | tok/s | llama.cpp | 比值 | matmul | attention | QuantInfer KiB | llama.cpp KiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.813 | 50.594 | 17.99 | 2825.9 ms，99.5% | 6.9 ms，0.2% | 3148944 | 1874016 |
| 2 | 5.325 | 80.234 | 15.07 | 1487.8 ms，99.0% | 6.6 ms，0.4% | 3185456 | 1868576 |
| 4 | 8.655 | 143.006 | 16.52 | 893.3 ms，98.0% | 10.3 ms，1.1% | 3185552 | 1868016 |
| 8 | 14.917 | 181.131 | 12.14 | 522.9 ms，97.4% | 6.2 ms，1.2% | 3185472 | 1875536 |

4 对 1 是 3.08 倍。这是 GGUF generate 的峰值 RSS，不是 QIF 加载对照。prompt 仍是 token `1,2,3,4,5`，所以生成的 id 很小。
