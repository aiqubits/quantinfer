# QIF v2.0 权重容器标准方案

状态：QIF v2.0-SINGLE、QIF2-PACKAGE、独立压缩 chunk 和 Ed25519 签名 envelope
已实现并冻结；跨语言 golden reader 与 benchmark 仍是验证工作。

QIF v1 已停止作为 QuantInfer 的目标格式。当前仓库中的 QIF v1 reader、writer 和
历史测试只保留为旧 artifact 的读取与迁移参考；新 GGUF 转换器写入 QIF2-SINGLE，
新 runtime 不得把 QIF v1 当作新版格式的兼容依据。

本文定义全新的 QIF v2.0 wire format。当前仓库实现 QIF2-SINGLE 与
QIF2-PACKAGE 两种 profile。文中的 **MUST**、**MUST NOT**、**SHOULD**、**MAY**
具有规范含义：MUST/MUST NOT 是互操作性要求，SHOULD 是默认实现要求，MAY 是可选能力。

## 1. 定位与边界

QIF v2 是面向推理部署的权重容器，目标是同时提供：

- 安全的无代码执行读取；
- 可验证的权重完整性；
- F32/F16/BF16/整数和块量化等物理编码；
- 文件级 mmap 和按 tensor/chunk 的范围读取；
- 单文件部署与可扩展分片部署；
- 逻辑 tensor 语义与物理存储布局分离；
- 小型 Rust/C/C++ runtime 可读取的最低依赖。

QIF v2 **不是**：

- PyTorch、TensorFlow、DCP 或 Orbax 的训练恢复格式；
- ONNX 这类包含执行图和 operator schema 的交换格式；
- TensorRT/CUDA/Metal 等硬件专用 executable；
- 任意 Python 对象或 pickle 的序列化容器。

SafeTensors 适合安全的框架间 tensor 交换，GGUF 适合 GGML 生态的单文件 LLM 推理，DCP/Orbax 适合分布式训练恢复。QIF v2 的目标是建立一个聚焦推理权重、量化语义、验证和按需读取的独立边界。这是设计目标，不是未经基准测试的性能承诺。

参考：

- [SafeTensors format](https://github.com/safetensors/safetensors)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [PyTorch Distributed Checkpoint](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/state_dict_saver.py)
- [ONNX external tensor data](https://onnx.ai/onnx/repo-docs/ExternalData.html)

## 2. 对当前 QIF v1 的事实审查

当前实现见 [qif/mod.rs](../../crates/quantinfer-format/src/qif/mod.rs)、[qif/reader.rs](../../crates/quantinfer-format/src/qif/reader.rs)、[qif/writer.rs](../../crates/quantinfer-format/src/qif/writer.rs) 和 [CPU model loading](../../crates/quantinfer/src/cpu/model.rs)。它具备 header checksum、tensor checksum、64 字节对齐、重复键拒绝、有限值校验和若干 GGML quant block 支持。

但 v1 的 wire assumptions 不适合继续扩展：

1. header 后是 manifest JSON、typed metadata 和按 manifest 顺序排列的 payload；没有独立 binary index。
2. descriptor 能推导 payload 大小，但不能表达 chunk、压缩、alias、外部 shard 或全局 shard coordinate。
3. `ModelSource::open_reader` 当前先读取整个文件，格式布局本身没有提供按需文件 reader API。
4. 量化编码映射集中在当前有限的 GGML 类型集合中，无法作为通用 encoding registry。
5. v1 的测试固定检查版本字段为 `1`；修改 v1 布局会产生隐式不兼容。

因此，新版必须使用 `version = 2`，不能通过添加未声明字段的方式修改 v1。

### 2.1 对上一版 v2 提案的修订

本方案明确修正了以下容易被误读为事实的内容：

- “比 SafeTensors/GGUF/DCP 更快或更轻”是待验证的设计目标，不是规范事实；
- 当前 QIF v1 没有 file-backed mmap API，v2 把它列为实现要求而不是现状；
- fixed header 的字段、大小和偏移固定为 144 字节；v2.0 reader 要求该值精确为 144，扩展头保留给后续版本；
- manifest 采用 QuantInfer v2.0 定义的确定性 JSON，metadata 采用确定性 QMB typed map，避免把可读性和热路径索引混为一谈，也避免 v2.0 强制引入 CBOR runtime 依赖；
- SHA-256 是 v2.0 必选 hash；BLAKE3 仍是后续可选 hash，Ed25519 envelope 已作为可选签名能力实现；
- chunk compression 支持独立 zstd/LZ4 chunk，`none` 仍是基础 codec；
- 单文件与分片包是两个 profile，PACKAGE reader 会验证全局 tensor map 和 shard coordinate；
- 训练 checkpoint 论文提供设计启发，但不把 optimizer、训练对象或增量恢复能力强行塞入 QIF 推理核心。

## 3. QIF v2.0 总体结构

QIF v2.0 冻结两种容器 profile：

- **QIF2-SINGLE**：一个文件，适合本地、边缘和单机服务。
- **QIF2-PACKAGE**：`qif.json`、`index.qifi`、`metadata.qmb` 和多个 `.qifd` shard，适合对象存储、大模型和并行读取。

两种 profile 使用同一套 manifest、tensor record、chunk record 和 hash 规则。差异只在 payload 的物理位置与 package manifest。

### 3.1 文件区域

QIF2-SINGLE 的区域顺序为：

```text
fixed header
canonical manifest
canonical metadata
binary index (string table, shape table, tensor records, chunk records)
optional signature envelope
aligned payload chunks
```

区域由 header 中的绝对 offset/length 指向；reader MUST NOT 依赖区域顺序推导未知区域。

### 3.2 固定 header

header 长度为 144 字节；下表字段按文件顺序排列，字段 offset 由前面字段大小累加得到。所有整数使用 little-endian。

| 字段 | 大小 | 约束 |
| --- | ---: | --- |
| magic | 4 | 固定为 `QIF\0` |
| version | 4 | 固定为 `2` |
| flags | 4 | profile、feature 和必需能力位 |
| header_size | 4 | v2.0 固定为 `144` |
| hash_algorithm | 4 | v2.0 必须支持 SHA-256；其他算法需能力协商 |
| index_version | 4 | 当前为 `1` |
| manifest_offset | 8 | 文件绝对偏移 |
| manifest_length | 8 | canonical manifest 长度 |
| metadata_offset | 8 | 文件绝对偏移 |
| metadata_length | 8 | canonical metadata 长度 |
| index_offset | 8 | binary index 起始偏移 |
| index_length | 8 | index header、string/shape table、tensor/chunk records 总长度 |
| payload_offset | 8 | payload 区域起始偏移 |
| payload_length | 8 | payload 区域长度 |
| root_hash | 32 | 对 canonical sections 和 chunk digest 的根校验 |
| signature_offset | 8 | 无签名时为 0；有签名时指向 108 字节 envelope |
| signature_length | 8 | 无签名时为 0；有签名时固定为 108 |
| reserved | 8 | 必须为 0，未来版本可扩展 |

未知的必需 feature bit MUST 拒绝；未知的 advisory bit MAY 忽略。reader MUST 检查所有区域的边界、重叠、整数溢出和文件 EOF。

QIF2-SINGLE v2.0 的 `flags` 固定为 `0`；codec 通过 chunk record 的 codec_id 表达，
签名通过 header 的 signature offset/length 表达。v2.0 reader 遇到非零 flags 必须拒绝。

## 4. Manifest 与 metadata

### 4.1 Manifest

manifest 使用 QuantInfer v2.0 定义的确定性 JSON 编码：由已验证的
`ModelManifest` 结构按固定字段顺序序列化为 UTF-8 JSON，reader 要求输入字节与
该序列化结果完全一致。v2.0 不声称实现 RFC 8785 的全部数值规范化规则；若未来
需要跨实现的 RFC 8785 互操作，必须单独冻结对应 canonicalization profile。JSON
主要用于检查、审计、转换和人工诊断，高频 tensor 定位由 binary index 完成。

manifest MUST：

- 拒绝重复 key；
- 拒绝非 UTF-8；
- 限制最大字节数、最大深度和最大字符串长度；
- 使用确定性 key 顺序和数字表示；
- 记录 `schema_version` 与 `model_id`；
- 记录 architecture、source revision、tokenizer、compatibility 和 tensor 列表；
- 每个 tensor 在 manifest 中记录 name 与完整 logical storage descriptor；稳定的
  `tensor_id`、shape table 和物理 encoding 由 binary index 记录。semantic role
  需要时通过受控 metadata key 表达，不能被 reader 当作存储布局。

manifest MUST NOT 保存任意可执行对象。

### 4.2 Metadata

metadata 使用 QuantInfer 定义的 QMB（QIF Metadata Binary）typed map，而不是
强制引入第三方 object serialization。QMB 的 key 按 UTF-8 编码字节排序，所有
map/array 使用 definite-length 编码，因此同一逻辑 metadata 只有一种 wire
表示。`qif inspect --json` 提供 JSON 展示；JSON 不是 payload wire format。

允许的值类型：

- UTF-8 string；
- unsigned/signed integer；
- finite float32/float64；
- boolean；
- bounded array；
- bounded map。

QMB v1 tags MUST be stable：`u32=4`、`i32=5`、`f32=6`、`bool=7`、
`string=8`、`array=9`、`u64=10`、`i64=11`、`f64=12`、`map=13`。数组元素
必须使用同一 tag；map key 必须是非空 UTF-8 string 并按编码字节排序。

每个整数 tag、array tag 和 map tag 使用 little-endian `u32`；整数和浮点值
随后使用对应 little-endian 固定宽度。string 使用 `u64 byte_length + UTF-8 bytes`；
array 使用 `u64 count + count 个带 tag 的 value`；map 使用 `u64 count + count
个 string key/value 对`。v2.0 reader 限制嵌套深度、项目数、字符串长度和 metadata
总长度，并拒绝 trailing bytes、重复 key、非有限浮点和未知 tag。

metadata MUST 设置最大深度、最大 map/array 项数和最大字符串字节数。metadata 中的 model/runtime hint 只能影响优化，不能改变 tensor 的正确解释。

## 5. Binary index

index 由固定头和四个确定性区域组成：

```text
index header
string table
shape table
tensor records
chunk records
```

### 5.1 Tensor record

index header 固定为 48 字节：`magic[4] = QIDX`、`index_version u32`、
`tensor_count u32`、`chunk_count u32`，以及 string table、shape table、tensor
record、chunk record 的四个 `u64` 字节长度。所有表都紧跟 index header，不能有
隐式尾部数据。

v2.0 tensor record 固定为 120 字节：

```text
tensor_id u64
name_ref u32, shape_ref u32
logical_dtype u32, physical_encoding u32, quantization_scheme u32, layout_id u32
codec_id u32, flags u32, chunk_start u32, chunk_count u32
raw_length u64, stored_length u64, data_offset u64
tensor_sha256[32]
alias_target u64 (all-ones means none)
reserved u32, reserved u32
```

shape table entry 为 `rank u32` 加 `rank` 个 `u64` dimension；string table entry
为 `length u32` 加 UTF-8 bytes。chunk record 固定为 72 字节：

```text
shard_id u32, codec_id u32
raw_offset u64, raw_length u64, stored_offset u64, stored_length u64
chunk_sha256[32]
```

每个 tensor record MUST 包含：

```text
tensor_id
name_reference
logical_dtype
physical_encoding
quantization_scheme
layout_id
rank
shape_reference
data_offset / chunk_table_reference
stored_length
raw_length
codec_id
checksum_reference
alias_target (optional)
flags
```

`logical_dtype`、`physical_encoding`、`quantization_scheme` 和 `layout_id` MUST 分开存储。`Q4_K` 不能同时充当 dtype、量化规则和物理布局名称。

tensor name MUST 是唯一的 UTF-8 字符串。`tensor_id` 在同一 artifact 中唯一且稳定。

### 5.2 Shape、stride 和 layout

rank 和 shape 使用 64 位无符号整数；stride 在 v2.0 不存储，后续 layout profile
不得复用当前 tensor record 的 reserved 字段。实现 MUST 限制最大 rank，防止恶意
shape 造成分配或乘法溢出。

QIF2-SINGLE v2.0 内建 layout：

```text
row_major
ggml_output_rows
```

新的 layout 必须使用新的 registry ID；不能通过复用旧 ID 改变含义。

### 5.3 Alias

如果多个逻辑 tensor 共享相同物理 payload，除第一个 tensor 外，其余 tensor MUST 使用 `alias_target`，不得重复写入相同 bytes。

## 6. Payload、chunk 和 codec

### 6.1 Chunk 规则

payload 由独立 chunk 组成。每个 chunk record MUST 包含：

```text
raw_offset
raw_length
stored_offset
stored_length
codec_id
chunk_hash
```

QIF2-PACKAGE 的 chunk record 还 MUST 包含 `shard_id`；QIF2-SINGLE 使用
固定的 `shard_id = 0`。

v2.0 默认 chunk alignment 为 4096 字节；实现 MAY 使用更大的 64 KiB 或 2 MiB 对齐。alignment 必须是 2 的幂，且不小于 64 字节。

无压缩 chunk MUST 可以通过 offset/length 直接读取；压缩 chunk MUST 可以独立解压和
独立校验，禁止使用跨 chunk 的连续压缩流。mmap 只适用于无压缩 chunk。

### 6.2 Codec

QIF2-SINGLE/QIF2-PACKAGE v2.0 core codec：

```text
0 = none       (MUST)
1 = zstd       (supported)
2 = lz4        (supported)
```

每个压缩 chunk 独立编码、独立解码和独立校验；reader 不使用跨 chunk 连续压缩流。
`QifV2Reader::tensor_data` 与 file/package reader 会按需解压当前 tensor 数据。

### 6.3 量化编码

量化描述 MUST 至少包含：

```text
scheme_id
block_size
group_size
quant_axis
scale_dtype
zero_point_dtype
packing_order
physical_layout
```

QIF2-SINGLE v2.0 实现固化 F32、F16、BF16 和当前 QuantInfer 使用的
Q4_0/Q4_K/Q5_K/Q6_K/Q8_0。FP8、INT8、INT4、NF4 需要在 descriptor、payload
大小和 golden fixtures 一起冻结后才能加入兼容集合。

## 7. 完整性与安全

### 7.1 Hash

SHA-256 是 v2.0 必选算法，以保持跨语言和现有 QuantInfer 生态的可用性。BLAKE3 可以作为后续可选算法，但不能成为 v2.0 的唯一依赖。

v2.0 的 hash 层级为：

```text
chunk hash = SHA-256(stored chunk bytes)
tensor hash = SHA-256(raw tensor bytes)
root hash = SHA-256(canonical manifest + canonical metadata + binary index)
```

reader 打开时至少验证 header、manifest、metadata、index 和 root hash；显式
`verify()` 必须读取并验证所有 tensor 与 payload chunk。v2.0 不定义命令行
`verify --fast/--full` 选项。

### 7.2 Signature

v2.0 定义固定 108 字节 Ed25519 envelope：`QSIG` magic、版本、算法 ID、公钥和
签名。签名覆盖 root hash；reader 验证 envelope，但信任根和密钥分发由发布系统负责。
COSE、证书链和密钥轮换不属于 v2.0 wire core。

### 7.3 禁止项

QIF v2 MUST NOT：

- 执行反序列化代码；
- 保存 pickle 或任意 Python object；
- 允许 metadata 路径穿越；
- 允许 unchecked allocation；
- 允许 NaN/Infinity 进入标准浮点 payload；
- 允许未知 mandatory encoding 被当作其他 dtype 读取。

## 8. QIF2-PACKAGE

QIF2-PACKAGE 由一个 package manifest、一个全局 binary index 和多个独立 shard 构成：

```text
qif.json
metadata.qmb
index.qifi
shard-00000.qifd
shard-00001.qifd
...
```

package manifest 为每个 shard 记录：

- 相对路径或受限 URI；
- 文件大小；
- shard hash；
- mesh axis、rank 和 shard coordinate；
- 所属 tensor 名称；
- package root hash（覆盖 root_hash 置空后的 canonical package manifest、metadata、global index 和 shard hashes）；
- model manifest 中的 global shape 与 tensor checksum。

路径 MUST 禁止 `..`、绝对路径和未授权 scheme。package 的发布必须使用临时目录、完整校验和原子 rename；未完成 package 不得被 reader 视为有效模型。

## 9. Runtime API

`quantinfer-format` 的内存、file-backed、mmap 和 package reader 已独立于 QIF v1；当前 API 为：

```rust
QifV2Reader::open(bytes)
QifV2Reader::manifest()
QifV2Reader::metadata()
QifV2Reader::tensor_info(name)
QifV2Reader::tensor_bytes(name)
QifV2Reader::read_range(name, range)
QifV2Reader::verify()
QifV2FileReader::open(path)
QifV2FileReader::read_range(name, range)
QifV2FileReader::verify()
QifV2MappedFile::open(path)
QifV2MappedFile::reader()
QifV2PackageReader::open(path)
QifV2PackageReader::read_range(name, range)
QifV2PackageReader::verify()
write_qif_v2_with_codec(..., QifV2Codec::Zstd|Lz4)
write_qif_v2_signed(..., signing_key)
```

API 要求：

- 内存 reader 打开时验证 header、manifest、metadata、index 和 root hash；
- `read_range` 只复制请求的 tensor 子范围；
- v2.0-SINGLE/PACKAGE 提供按需 file-backed range read，并提供只读 `QifV2MappedFile` 映射入口；
  mmap 的 unsafe 操作被隔离在单独模块，调用者仍必须先通过同一 QIF2 reader 校验；
- reader 可选择调用 `verify` 执行全 payload 校验；
- 文件、内存 buffer 和 package shard 使用同一逻辑 tensor/chunk 校验规则；
- Rust/C/C++ binding 不依赖 PyTorch、Python、ONNX Runtime 或 llama.cpp。

## 10. 性能目标与验收边界

以下是必须验证的工程目标，不是未经测试的既定优势：

1. file-backed index open 不读取 payload（当前 `QifV2FileReader::open` 已满足）；
2. 只读取一个 tensor 时，不触碰其他 tensor 的 payload（当前 file-backed `read_range` 已满足）；
3. 无压缩路径能返回 mmap/range view（当前 `QifV2MappedFile` 与 file-backed range read 已满足）；
4. 压缩路径只解压当前请求的 tensor/chunk（当前 codec reader 已满足）；
5. QIF2-SINGLE 与 QIF2-PACKAGE 对相同 tensor 产生一致逻辑值（当前 package tests 已覆盖）；
6. package 通过全局 index 改变 shard 分配而不改变 tensor 编码（当前 writer 按 tensor 分片，reshard planner 仍需额外验证）；
7. root/full verification 在损坏、截断、重叠、越界和错误 codec 上稳定失败；
8. Rust reader 与当前 C/Python golden readers 对同一 uncompressed fixture 得出一致 header、index 和 payload bytes；
9. 与 SafeTensors/GGUF 的 benchmark 必须报告：open latency、first-tensor latency、RSS、page faults、selected-layer read bandwidth、full-load time 和验证吞吐；
10. 不得把“比 SafeTensors 快多少”写入规范，除非由可复现基准证明。

## 11. 从当前仓库迁移

迁移顺序：

1. 新增 `qif-v2` module，禁止在 v2 module 复用 v1 header parser（已完成）。
2. 将当前 `qif` module 标记为 legacy-only，新的 GGUF 转换器写入 QIF2（已完成）。
3. 增加 QIF2 binary index、内存 reader、single-file writer（已完成）。
4. file-backed reader、mmap/range view、QIF2 package/shard writer 已完成。
5. QIF2 的 Rust fixture、dependency-free Python reader 和 C wire reader smoke test 已完成；fuzz target、更多 malformed corpus 和 benchmark 仍是持续验证工作。
6. 新 CLI 默认输出 QIF2；默认 runtime 拒绝旧 v1，旧 v1 module 仅保留在格式 crate 内供历史 fixtures 使用。
7. GGUF → QIF2 已由 `from-gguf` 实现；SafeTensors 当前通过既有
   SafeTensors → GGUF → QIF2 两步路径转换，直接 SafeTensors → QIF2 仍待单独的
   descriptor/metadata 映射冻结后实现。
8. 更新 `docs/architecture/design-decisions.md`、`docs/architecture/mvp-qwen3.md`、`docs/rfcs/README.md`，明确 v1 已废弃、v2 是唯一新格式（已完成）。

## 12. 设计结论

QIF v2 的实际价值不是声称全面击败 SafeTensors、GGUF 或 DCP，而是明确解决它们之间的组合空白：

- 继承 SafeTensors 的无代码执行和简单 tensor 读取边界；
- 继承 GGUF 的 self-describing metadata、量化和单文件推理思路；
- 吸收 DCP/Orbax/Universal Checkpointing 的 global shape、offset、shard 和 resharding 元数据；
- 吸收 fastsafetensors、DataStates-LLM、TierCheck 的批量、异步、分层和按需 I/O 思路；
- 保持 QuantInfer 自有的 logical descriptor、semantic tensor role 和 backend capability 边界。

因此，QIF v2.0 的成功标准是：**同一套可验证权重语义，能在单文件或 package
shard 中通过 mmap、file-backed range read、独立 codec 解压和完整校验稳定加载。**
性能优劣仍必须由可复现 benchmark 证明，不能由格式设计直接推出。

## 13. 参考资料

- [SafeTensors specification](https://github.com/safetensors/safetensors)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [PyTorch Distributed Checkpoint](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/state_dict_saver.py)
- [ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData.html)
- [Universal Checkpointing](https://arxiv.org/abs/2406.18820)
- [ByteCheckpoint](https://www.usenix.org/system/files/nsdi25-wan-borui.pdf)
- [fastsafetensors](https://arxiv.org/abs/2505.23072)
- [DataStates-LLM 2026](https://arxiv.org/abs/2601.16956)
- [TierCheck 2026](https://arxiv.org/abs/2605.17821)
- [LayerCheck 2026](https://arxiv.org/abs/2609.27193)
- [Lossless Compression for LLM Tensor Incremental Snapshots](https://arxiv.org/abs/2505.09810)
