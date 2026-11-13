# QI-002 注意事项

- `AGENTS.md` 不是新流程。developer guide 改了，这份文件要跟着改。
- 作者的提交政策仍是直接进 main。这份文件复述了它，没有改成 fork 上的分支政策。
- Runbook 仍然只覆盖 CLI 单序列。server 的批处理和流式响应写在 `apps/README.md`，不要把那些步骤抄进 Runbook。
- 对照 commit `ee63330` 描述的是上游 CLI。本分支后来加了 `--threads`，Runbook 的示例没有用这个参数。
