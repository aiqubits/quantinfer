# QIF format status

The public artifact and API name is **QIF**. Wire-version details remain
internal to the implementation and documentation. Historical QIF v1 is not
accepted by the default runtime; its wire description is preserved in
[qif-v1-legacy.md](qif-v1-legacy.md) only for migration and fixture analysis.

The current wire format is QIF v2.0. Its normative design is documented in
[qif-v2-final-plan.md](qif-v2-final-plan.md).

QIF-SINGLE 与 QIF-PACKAGE 的格式库 reader/writer、独立压缩 codec、mmap、
file-backed 校验器和确定性 reshard planner 已有实现。GGUF/SafeTensors 转换器当前
输出 SINGLE；Ed25519 envelope 也仅适用于 SINGLE。默认模型加载器尚未直接接入
原生 `qif.json` PACKAGE，不能将格式库支持等同于所有推理入口均支持。
对外使用 `quantinfer_format::qif::{Reader, FileReader, PackageReader,
MappedFile, Codec}` 等稳定名称。`MappedFile::open` 为 unsafe 构造函数：
调用方须保证底层文件在映射及其所有
借用 view 存活期间不被修改/截断。不能提供这一保证时使用安全的 `FileReader`。
[Rust/Python/C golden 验证](qif-conformance.md)有明确的固定 profile 边界；
[跨格式测量](../../benches/README.md)按进程隔离，
记录加载策略、选中 tensor、dtype、范围 hash 和未控制的缓存状态，不能直接推出
格式的固有性能优劣。QIF 带有 seeded malformed corpus 和带 ASan/覆盖率的
[CI fuzz smoke](../../fuzz/README.md)，持续长时 fuzzing 仍属于后续验证工作。Legacy v1 wire code
仅保留在 format crate 内供历史 fixtures 使用，默认 runtime 和转换路径均拒绝或不生成 v1。
