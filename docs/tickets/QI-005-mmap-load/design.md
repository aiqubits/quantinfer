# QI-005 设计

映射调用放在新 crate `quantinfer-map`。`MappedFile::map` 是唯一的 `unsafe` 调用，因为 `memmap2` 0.9 把 `Mmap::map` 标成了 unsafe：映射还活着的时候把文件截短是未定义行为。format、core、kernel 和标量 CPU 继续 `forbid(unsafe_code)`。`memmap2` 用 0.9.12，和 candle 已经锁住的版本相同。

format 里的文件字节是 `ArtifactBytes`。它用 `Arc` 包住两种后端：内存 `Vec<u8>`，或者 `MappedFile`。`GgufReader` 仍然可克隆，克隆只增加引用计数。对外只暴露 `&[u8]`。

`GgufReader::open` 和 `open_artifact` 继续接受 `Vec<u8>`。新增 `open_artifact_path`：先 `canonicalize`，再用 `O_NOFOLLOW` 打开普通文件，映射后核对映射长度和当时的文件长度。魔数是 QIF 时走原来的 QIF 解析，否则走 GGUF。GGUF 在解析完张量表之后检查每个张量的区间落在文件内，短了就返回 `GGUF tensor data is truncated`。

CLI 和 CPU 模型加载走这条路径。模型包先映射，再用清单里的 SHA-256 核对映射内容，然后才把 reader 交出去。`read_verified` 仍保留给需要整份字节的调用。

解码没有改。`tensor_q8_0`、`tensor_q4_0` 和其他张量读取仍把块复制进运行时张量。reader 在模型加载完成后丢弃，映射随之释放。常驻内存仍是解码后的权重。

调用约定写在 `MappedFile::map` 上：文件在映射存活期间不能变短。打开时已经短于张量区间的文件会报错。并发截短无法在 safe Rust 里变成一个可检查的错误。
