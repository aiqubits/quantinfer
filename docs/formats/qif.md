# QIF format status

QIF version 1 is retired and is not the target format for new QuantInfer
artifacts. The historical wire description is preserved in
[qif-v1-legacy.md](qif-v1-legacy.md) only for migration and fixture analysis.

The new format is QIF v2.0. Its normative design is documented in
[qif-v2-final-plan.md](qif-v2-final-plan.md).

QIF2-SINGLE v2.0 的 reader、writer、GGUF 转换路径和内存/file-backed 校验器已在
`quantinfer-format::qif_v2` 实现。当前实现只接受无压缩单文件 profile；QIF2-PACKAGE、
压缩 codec、签名和跨语言 golden fixtures 仍是后续兼容工作。Legacy v1 wire code
仅保留在 format crate 内供历史 fixtures 使用，默认 runtime 和转换路径均拒绝或不生成 v1。
