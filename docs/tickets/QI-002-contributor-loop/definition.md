# QI-002 Harness 落盘

## 目标

把只存在于作者会话里的协作约定写成仓库文件，并改掉已经和代码矛盾的 Runbook，
让下一次 clone 能按同一套规程工作。

## 约束

- `AGENTS.md` 只写这个仓库特有的边界、完工定义、提交循环和本地预检。
- 条文和 `docs/developer-guide/README.md` 一致。两边冲突时以 developer guide 为准，并同时改 `AGENTS.md`。
- 文档修正对照当时的代码，不凭印象改。

## 上下文

仓库里没有 `AGENTS.md`、`CLAUDE.md` 或 `.cursor` 规则。规程散在 developer guide、
架构文档和 CI 里。`docs/user-guide/qwen3-production-runbook.md` 曾把多请求调度
写成不支持，但 `quantinfer-server` 已经有 `--max-batch-sequences` 和
`--max-batch-tokens`。

## 关联

没有代码前置。它是后面每张票的软前置。QI-008 管版本发布，这张票只管规程落盘。

## 非目标

- 不改运行时行为。
- 不重写全部文档。只修和代码矛盾的句子，并指出 agent 从哪一份文件读规则。

## 验收

- `AGENTS.md` 写明依赖方向、量化布局的完工定义、直接提交到 main 的循环，以及五条本地预检。
- Runbook 里多请求和在线平台两处改成和 server 一致，并标上对照的 commit。
- 工作区测试、fmt、clippy 保持通过。这张票不改变代码行为。
