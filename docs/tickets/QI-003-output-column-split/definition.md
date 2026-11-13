# QI-003 Decode 按输出列切分

## 目标

单 token decode 只有一行输入，按行切分等于没有并行。量化 matmul 在行数少于线程数时改为按输出列切分，让 `--threads` 对 decode 产生加速。

## 约束

- `threads = 1` 时结果和标量路径逐位相同。
- 只改并行方式，不改量化块的内存布局。
- 用 `std::thread::scope`。不引入 rayon，不加常驻线程池。
- 标量 CPU 后端继续 `forbid(unsafe_code)`。

## 上下文

QI-001 在 M4 Pro 上看到 SmolLM2-135M Q8_0 的 decode 在 1、2、4、8 线程都是大约 13.2 tok/s。
`matmul_rows` 按输入行切分，decode 的行数是 1，工作线程被压到 1。GGML 的 output-row 布局走 `matmul_output_rows`，当时整段是串行的。llama.cpp 按输出列做 matvec，同机大约 349 到 543 tok/s。

## 关联

依赖 QI-001 的对照和根因。为 QI-006 留下并行基线。

## 非目标

- 不做 NUMA 绑核。
- 不改 attention 的并行，那是 QI-006。
- 不做常驻线程池。权重是带生命周期的借用，常驻线程要把借用跨过 scope，标量后端不允许为此使用 unsafe。

## 验收

- SmolLM2-135M Q8_0 在 4 线程下的 decode tok/s 至少是单线程的 2 倍。
- 固定输入下，生成的 token 和单线程逐个相同。
- 1/2/4 的曲线写入 `benches/llama-cpp`，并更新 `benches/README.md`。
- prefill 不回退。
