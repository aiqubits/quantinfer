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

并入 `feature/cpu-decode` 时，Runbook 的对照 commit 改成 `ee63330`。`AGENTS.md` 增加了 `quantinfer-map` 这一句，因为这条分支已经有这个 crate。并入后没有单独再跑一遍全工作区测试。同一条分支上的 CPU 与 facade 单测在合并 QI-001 和 QI-003 之后通过，见 QI-003 的验证记录。
