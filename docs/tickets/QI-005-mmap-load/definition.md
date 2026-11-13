# QI-005 映射 GGUF 和 QIF

## 目标

按路径打开 QIF 和 GGUF 时映射文件，不再把整个文件读进一块 `Vec`。票面还要求用文件字节直接构造量化张量，并让 Qwen3-0.6B Q4_0 的加载时间和峰值 RSS 下降，RSS 目标是 30% 以上。

## 约束

- `memmap` 类型不出 format 层。core 和 runtime 看不到它。
- 打开时已经截断，或者校验失败，必须返回错误。
- 内存里的 `open(Vec<u8>)` 保留，测试和写入路径继续用它。

## 上下文

QIF 读取器已经能在借用的字节上校验并返回张量区间。加载路径此前用 `read_file_nofollow` 把文件读进 `Vec`，再把每个张量解码成拥有所有权的块。`memmap2` 只是 candle 的传递依赖。

## 关联

依赖 QI-004。加载时间和 RSS 原计划对照 QI-001。QIF 规范冻结（QI-007）要用这里的加载行为做论据。

## 非目标

- 不做缺页调度或按需懒加载。
- 不改 GGUF 或 QIF 的文件格式。
- 不处理网络文件系统。

## 验收

- Qwen3-0.6B Q4_0 从 QIF 加载时，`model_load_ms` 和峰值 RSS 下降，RSS 目标 30%。
- 固定 seed 下，生成 token 和原来的路径一致。
- 截断和校验失败有测试。
- `benches/baselines` 更新加载指标，`check-regressions` 通过。
