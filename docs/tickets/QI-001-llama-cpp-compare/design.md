# QI-001 设计

对照脚本是 `benches/llama-cpp-compare.py`。它按 `benches/real-model.sh` 的生成约定调用
`quantinfer-cli generate`，再调用钉死的 `llama-bench`。

双方约定：

- prompt 是 token `1,2,3,4,5`
- 贪心生成 8 个 token，temperature 0
- 上下文 64
- llama.cpp 构建钉在 `0ec4027a2a0183de87c007069adddc06e665f7de`
- `llama-bench` 必须带 `-ngl 0`。macOS 上省略它会走 Metal，线程曲线没有意义
- llama-bench 的 stderr 丢掉，JSON 从第一个 `[` 开始解析

报告字段沿用 real-model 的 key=value 口径，外加模型文件名、GGUF SHA-256、
CPU 型号、架构、llama.cpp commit，以及每一档的 prefill、decode 和两者的比值。
比值是 llama.cpp 的 decode tok/s 除以 QuantInfer 的 profile decode tok/s。

CLI 增加 `--threads N`。它和 `--inference-profile` 互斥，因为 profile 自己带线程数。
非 CPU 后端拒绝这个参数。`0` 是错误。默认仍是 1，这样旧的 real-model 基线不用重录。
`benches/real-model.sh` 用 `THREADS` 环境变量把这个值传进去，缺省也是 1。

CI 工作流是 `.github/workflows/llama-cpp-compare.yml`。它按 `gguf-interop.yml`
的方式检出同一颗 llama.cpp，编 `llama-bench` 和 `llama-quantize`，再编
release CLI，下载 SmolLM2 F16，校验 SHA-256，量化成 Q8_0，然后跑 1、2、4 线程。

`--profile` 在 decode 循环里分别累计 matmul、attention 和 sampling。其余时间用
decode 墙钟减去这三项。计时默认关闭，只在 `--profile` 时包住调用。

macOS 的峰值 RSS 读 `getrusage` 的 `ru_maxrss`（字节，再换成 KiB）。Linux 仍读
`VmHWM`。llama.cpp 的 RSS 从 `/usr/bin/time` 的最大驻留集解析。

`benches/check-llama-cpp-report.py` 检查线程、RSS、分项时间和 token 列表。
对照 JSON 继续放在 `benches/llama-cpp/`。`check-regressions.py` 会扫
`benches/baselines/` 里的全部 JSON，而它只认 real-model 的 key=value 报告，
所以对照报告不进那个目录。

列切分之前的平线来自按输入行切分。那部分结论交给 QI-003。分项计时是在列切分之后的 release 上补的。
