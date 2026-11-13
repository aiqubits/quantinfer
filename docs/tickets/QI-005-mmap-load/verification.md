# QI-005 验证

在 `77db1f3` 上跑过：

```sh
cargo test --workspace --locked
cargo clippy --locked -p quantinfer-map -p quantinfer-format -p quantinfer --all-targets -- -D warnings
```

工作区测试通过。新增的测试是：

- `quantinfer-map` 把写入的字节映射回来。
- GGUF 路径打开后，F32 张量和内存中的 `open` 相等。砍掉最后一个字节后打开失败，错误含 `truncated`。
- QIF 路径打开后，`norm` 和 `weight` 的字节与写入内容相等。截断失败。翻转头部一个字节后，错误含 `header checksum mismatch`。

用 debug CLI 对 `/tmp/qi-bench/smollm2-q4_0.gguf` 做了一次冒烟：

```text
prompt_tokens=[1, 2, 3, 4, 5]
generated_tokens=[7676, 271, 260, 198, 1195, 390, 4269, 284]
```

进程正常退出。debug 的 tok/s 不是性能数据。没有和映射前的同一构建做逐 token 对照，也没有测加载 RSS。
