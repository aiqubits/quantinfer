# QIF format status

The public artifact and API name is **QIF**. Wire-version details remain
internal to the implementation and documentation. Historical QIF v1 is not
accepted by the default runtime; its wire description is preserved in
[qif-v1-legacy.md](qif-v1-legacy.md) only for migration and fixture analysis.

The current wire format is QIF v2.0. Its normative design is documented in
[qif-v2-final-plan.md](qif-v2-final-plan.md).

QIF-SINGLE 与 QIF-PACKAGE 的格式库 reader/writer、独立压缩 codec、mmap、
file-backed 校验器和确定性 reshard planner 已有实现。GGUF/SafeTensors 转换器可用
`--qif-profile single|package`、`--qif-codec none|zstd|lz4` 选择发布形式；PACKAGE
可指定 `--max-shard-bytes N`。Ed25519 envelope 仍仅适用于 SINGLE。
默认模型路径现在接受原生 `qif.json` PACKAGE 目录，并以只读控制区打开、按 tensor
读取的方式加载 SINGLE/PACKAGE。typed decoder 不缓存原始权重，显式借用字节接口
按物理 tensor 缓存并共享 alias。Dense 模型构造仍会解码所有需要的 Core 权重。
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

```sh
# Publish a native directory from GGUF (omit --qif-profile for SINGLE).
quantinfer-convert from-gguf --input model.gguf --output model-qif \
  --qif-profile package --qif-codec zstd --max-shard-bytes 16777216
quantinfer-cli generate --input model-qif --tokens 1,2,3 --max-new-tokens 8
```

The Rust entry point `PackedTensorReader::open_path` accepts a SINGLE file or
native PACKAGE directory. `tensor_data` returns a transient `Cow` and validates
the selected tensor's raw SHA-256 and finite values. `tensor_bytes` retains a
shared cache to satisfy its borrowed-slice lifetime; `cached_tensor_bytes`
reports that cache, counting aliases once. `verify_qif` streams all native
payload/shard checks without filling it. `manifest.json` model packages retain
their separately verified owned-file snapshot path. Neither API implies a
hard process-memory bound or layer-level out-of-core execution.
