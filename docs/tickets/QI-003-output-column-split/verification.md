# QI-003 验证

记录文件：`benches/llama-cpp/apple-m4-pro-smollm2-q8_0-column-split.json`。

和 QI-001 同一台 Apple M4 Pro、同一个 `smollm2-q8_0.gguf`、同一个 llama.cpp
`0ec4027a2`、同一套 prompt 和生成长度。这一轮的 CLI 带列切分。

| 线程 | QuantInfer decode tok/s | llama.cpp decode tok/s | 比值 | QuantInfer prefill ms |
| --- | --- | --- | --- | --- |
| 1 | 19.511 | 424.741 | 21.77 | 257.409 |
| 2 | 28.432 | 532.600 | 18.73 | 149.151 |
| 4 | 39.527 | 609.328 | 15.42 | 87.885 |

4 线程相对 1 线程是 39.527 / 19.511 = 2.03 倍。prefill 从 257 ms 降到 88 ms。
同一次协议里，1 线程和 4 线程打出的 token 都是
`6395, 1526, 30, 198, 198, 504, 808, 282`。JSON 里只存了生成个数 8，token 序列在当次运行输出里。

更早一次冷启动大约是 1.87 倍。隔离的 576×576 release matmul 能过 2 倍。随后三次成对运行大约是 2.03、1.96 和 2.06 倍。入库的是 2.03 这一次。

并入 `feature/cpu-decode` 之后，debug 单测通过：

```sh
cargo test --locked -p quantinfer --lib
cargo test --locked -p quantinfer-backend-cpu --lib
```

其中包含 `resolves_backend_with_an_explicit_thread_count` 和
`threaded_quantized_output_columns_match_scalar_reference`。release 缩放测试
因为 `debug_assertions` 没有在这次 debug 运行里执行。
