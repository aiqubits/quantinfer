# QIF format status

QIF version 1 is retired and is not the target format for new QuantInfer
artifacts. The historical wire description is preserved in
[qif-v1-legacy.md](qif-v1-legacy.md) only for migration and fixture analysis.

The new format is QIF v2.0. Its normative design is documented in
[qif-v2-final-plan.md](qif-v2-final-plan.md).

QIF2-SINGLE 与 QIF2-PACKAGE v2.0 的 reader、writer、GGUF 转换路径、独立压缩
codec、Ed25519 签名、mmap 和 file-backed 校验器已在 `quantinfer-format::qif_v2`
实现。Rust/Python/C 基线 reader 已加入验证路径，完整 benchmark 和 fuzz corpus
仍是验证工作。Legacy v1 wire code
仅保留在 format crate 内供历史 fixtures 使用，默认 runtime 和转换路径均拒绝或不生成 v1。
