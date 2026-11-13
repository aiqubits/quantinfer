# QI-002 验证

第一次核对发生在工作区 `quantinfer-qi-002`，基线是 `7c0cd4f`。当时跑过：

```sh
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo run --locked -p quantinfer-xtask -- bench
git diff --check
```

这些命令在那个工作区通过。那次改动只有 `AGENTS.md`、developer guide 的指针和 Runbook 三处句子。

并入 `feature/cpu-decode` 之后，在包含 QI-004、QI-005 和列切分的树上又跑了一次：

```sh
cargo test --workspace --locked
cargo run --locked -p quantinfer-xtask -- bench
```

工作区测试通过。`xtask bench` 退出码是 0。它在 macOS 上把 `rss_kb_before` 和 `rss_kb_after` 打成 0，因为那条路径仍只读 Linux 的 `/proc`。这不是 Runbook 修正的验收项，但不要把这个 0 当成内存测量。
