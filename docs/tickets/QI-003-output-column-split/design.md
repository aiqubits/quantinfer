# QI-003 设计

切分在 `backends/cpu` 的标量后端里，入口是 `map_output_columns`。

调用方给出输出列数和一个闭包。闭包计算 `[column_start, column_end)` 这一条，按行主序返回 `rows * width` 个 `f32`。线程数取 `threads.max(1).min(columns)`。多个线程用 `std::thread::scope` 启动，闭包先变成引用再分给每个 worker，避免第一次 spawn 把闭包移走。每个 worker 写入自己的列区间，最后按列拼回完整输出。每个列内部的求和顺序保持不变，所以单线程和原来逐位相同。

两条路径：

- 行主序的量化 matmul：行数不少于线程数时，仍按行切。行数更少时，改按列切。单 token decode 走这条。
- GGML output-row 布局（`matmul_output_rows`，以及 SIMD 的 Q8 和 Q4 output-row）：总是按列切。这是外部 `llama-quantize` 产物的 decode 路径。

块数据按列区间切片后借给 worker。闭包和打包函数标成 `Sync`。溢出用 `checked_mul`，不用饱和乘法。worker panic 的信息是 `CPU worker panicked`。

测试有两层。一层是小矩阵：1 行和 5 行、70 个输出列（不能被 4 整除）、Q8 output-row、Q4 output-row 和行主序 Q8，多线程结果和标量逐位相等。另一层只在 release 编译时跑：把 `[1, 576] x [576, 576]` 的 Q8 output-row 重复 40 次，要求 4 个 worker 的耗时满足 `4 线程 * 1.5 < 单线程`。2 倍这条线留给真实模型脚本，避免 CI 抖动。
