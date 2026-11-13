# 设计

## 对照

`benches/llama-cpp-compare.py` 按 real-model 的约定跑 `quantinfer-cli generate`：token `1,2,3,4,5`，贪心 8 个 token，上下文 64。llama.cpp 钉在 `0ec4027a2a0183de87c007069adddc06e665f7de`，`llama-bench` 必须带 `-ngl 0`。

CLI 增加 `--threads N`。它和 `--inference-profile` 互斥，非 CPU 后端拒绝它，`0` 是错误。默认 1，旧基线不用重录。

`--profile` 在 decode 循环里累计 matmul、attention 和 sampling。其余时间用 decode 墙钟减去这三项。不计 prefill。计时默认关闭。

macOS 峰值 RSS 用 `getrusage` 的 `ru_maxrss`（字节再换成 KiB）。Linux 仍读 `VmHWM`。llama.cpp 的 RSS 从 `/usr/bin/time` 解析。

`benches/check-llama-cpp-report.py` 检查线程、RSS、分项时间和 token 列表。报告放在 `benches/llama-cpp/`。

## 列切分

`backends/cpu` 的 `map_output_columns` 把输出列划给 `std::thread::scope`。闭包先变成引用再分给 worker。每个列内部的求和顺序不变，所以单线程逐位相同。

行主序量化 matmul 在行数少于线程数时按列切，否则仍按行切。GGML output-row 总是按列切。溢出用 `checked_mul`。常驻线程池排除掉了：scoped 借用的权重不能交给长寿命线程，标量后端不允许为此使用 unsafe。

## 视图

F32 存储改为 `Arc<Vec<f32>>`。行主序 `reshape` 只增加引用计数。stride 不是行主序时返回 `reshape requires a contiguous row-major tensor`。对共享存储写入会 copy-on-write。

`TensorView` 不再是 `Copy`。它带 `offset`、`row_slice`、`transpose` 和 `narrow`。连续视图才能 `as_f32`。越界和错误秩走 `get_f32` 的错误，不 panic。量化张量的存储没改。

## 映射

`quantinfer-map::MappedFile::map` 是唯一调用 `memmap2::Mmap::map` 的地方。`memmap2` 用 0.9.12，和 candle 已锁的版本相同。format 用 `ArtifactBytes` 包住内存 `Vec` 或映射，对外只给 `&[u8]`。

`open_artifact_path` 先 `canonicalize`，再 `O_NOFOLLOW` 打开普通文件，并核对映射长度。GGUF 在解析后检查每个张量区间落在文件内。模型包在映射上核对清单 SHA-256。`open(Vec<u8>)` 保留给测试和写入。

解码仍把块复制进运行时张量。reader 丢掉之后映射可以释放。映射还活着时把文件截短是未定义行为。打开时已经短于张量区间的文件会报错。

## 规程

`AGENTS.md` 复述 developer guide。两边不一致时以 developer guide 为准，并在同一次改动里更新 `AGENTS.md`。Runbook 只覆盖 CLI 单序列。多请求调度指向 `quantinfer-server` 和 `apps/README.md`。对照 commit 是 `ee63330`。本分支后来加的 `--threads` 没有写进 Runbook 示例。
