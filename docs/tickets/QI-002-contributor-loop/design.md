# QI-002 设计

`AGENTS.md` 放在仓库根目录。它复述 developer guide，不另起一套流程。开头写明：
两份文件不一致时，以 developer guide 为准，并在同一次改动里更新 `AGENTS.md`。

写进去的内容是：

- 依赖只能单向，从应用到 facade，再到 runtime、quant、model、format，再到 core 和 kernel，再到后端。
- core、kernel 和标量 CPU 后端保持 safe Rust，不依赖厂商 SDK、HTTP、Candle 或 WASI-NN。
- QIF 在规范和一致性测试稳定之前保持实验状态。MoE 和 sliding window 不支持。
- 一种新的量化布局要同时带上描述符、非法输入的校验错误、合成数据的往返、运行时分发测试，以及在有外部参考时的数值门禁。
- 提交直接进 main，一次一个可独立构建的改动。CI 失败用后续提交修，不改已推送的历史。
- 本地预检五条：fmt、`cargo check --locked`、`cargo test --locked`、`xtask bench`、clippy `-D warnings`。另外跑 `git diff --check`。模型一致性会下载钉死的权重，只有改到那条路径才跑。

developer guide 的第一段指向 `AGENTS.md`。

Runbook 只改三处。开头标上对照 commit。支持范围表下面把多请求调度指到 `quantinfer-server` 和 `apps/README.md`，并声明本文仍是 CLI 单序列。第 9 节不再把在线平台写成尚未实现的第二版。server 已有排队、连续批处理和流式响应，这些不在本文流程里，也不等于完整的生产平台。
