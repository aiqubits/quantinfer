# QIF format status

The public artifact and API name is **QIF**. Wire-version details remain
internal to the implementation and documentation. Historical QIF v1 is not
accepted by the default runtime; its wire description is preserved in
[qif-v1-legacy.md](qif-v1-legacy.md) only for migration and fixture analysis.

The current wire format is QIF v2.0. Its normative design is documented in
[qif-v2-final-plan.md](qif-v2-final-plan.md).

QIF2-SINGLE 与 QIF2-PACKAGE v2.0 的 reader、writer、GGUF/SafeTensors 转换路径、独立压缩
codec、Ed25519 签名、mmap、file-backed 校验器和确定性 reshard planner 已在
内部实现中。对外使用 `quantinfer_format::qif::{Reader, FileReader, PackageReader,
MappedFile, Codec}` 等稳定名称。Rust/Python/C 基线 reader 已加入验证路径；跨格式
性能对比可通过 `quantinfer-xtask qif-bench --qif ... --safetensors ... --gguf ...`
复现；QIF2 reader 已带有 seeded malformed corpus 和 CI fuzz smoke test，持续长时
fuzzing 仍属于后续验证工作。Legacy v1 wire code
仅保留在 format crate 内供历史 fixtures 使用，默认 runtime 和转换路径均拒绝或不生成 v1。
