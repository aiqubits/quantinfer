# QIF v2.0 权重容器标准方案

状态：设计冻结候选，尚未实现。

QIF v1 已停止作为 QuantInfer 的目标格式。当前仓库中的 QIF v1 reader、writer、GGUF 转换器和测试只保留为历史参考与迁移输入；新 writer 不得继续生成 QIF v1，新 runtime 不得把 QIF v1 当作新版格式的兼容依据。

本文定义全新的 QIF v2.0 wire format。文中的 **MUST**、**MUST NOT**、**SHOULD**、**MAY** 具有规范含义：MUST/MUST NOT 是互操作性要求，SHOULD 是默认实现要求，MAY 是可选能力。

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
- fixed header 的字段、大小和偏移已固定为可计算的 144 字节基础头；
- manifest 采用 canonical JSON，metadata 采用 canonical CBOR，避免把可读性和热路径索引混为一谈；
- SHA-256 是 v2.0 必选 hash；BLAKE3、Ed25519/COSE 不作为 v2.0 的强制依赖；
- chunk compression 是可选能力，`none` 是唯一必须支持的 codec；
- 单文件与分片包是两个 profile，不声称所有运行时都支持分片或 resharding；
- 训练 checkpoint 论文提供设计启发，但不把 optimizer、训练对象或增量恢复能力强行塞入 QIF 推理核心。

## 3. QIF v2.0 总体结构

QIF v2 允许两种容器 profile：

- **QIF2-SINGLE**：一个文件，适合本地、边缘和单机服务。
- **QIF2-PACKAGE**：一个 package manifest 加多个 data shard，适合对象存储、大模型和并行读取。

两种 profile 使用同一套 manifest、tensor record、chunk record 和 hash 规则。差异只在 payload 的物理位置与 package manifest。

### 3.1 文件区域

QIF2-SINGLE 的区域顺序为：

```text
fixed header
canonical manifest
canonical metadata
string table
tensor index
chunk index
optional signature envelope
aligned payload chunks
```

区域由 header 中的绝对 offset/length 指向；reader MUST NOT 依赖区域顺序推导未知区域。

### 3.2 固定 header

header 的基础长度为 144 字节；下表字段按文件顺序排列，字段 offset 由前面字段大小累加得到；`header_size` MAY 大于 144，用于向前兼容扩展。所有整数使用 little-endian。

| 字段 | 大小 | 约束 |
| --- | ---: | --- |
| magic | 4 | 固定为 `QIF\0` |
| version | 4 | 固定为 `2` |
| flags | 4 | profile、feature 和必需能力位 |
| header_size | 4 | `>= 144`，且不超过实现上限 |
| hash_algorithm | 4 | v2.0 必须支持 SHA-256；其他算法需能力协商 |
| index_version | 4 | 当前为 `1` |
| manifest_offset | 8 | 文件绝对偏移 |
| manifest_length | 8 | canonical manifest 长度 |
| metadata_offset | 8 | 文件绝对偏移 |
| metadata_length | 8 | canonical metadata 长度 |
| index_offset | 8 | binary index 起始偏移 |
| index_length | 8 | string/tensor/chunk index 总长度 |
| payload_offset | 8 | payload 区域起始偏移 |
| payload_length | 8 | payload 区域长度 |
| root_hash | 32 | 对 canonical sections 和 chunk digest 的根校验 |
| signature_offset | 8 | 没有签名时为 0 |
| signature_length | 8 | 没有签名时为 0 |
| reserved | 8 | 必须为 0，未来版本可扩展 |

未知的必需 feature bit MUST 拒绝；未知的 advisory bit MAY 忽略。reader MUST 检查所有区域的边界、重叠、整数溢出和文件 EOF。

## 4. Manifest 与 metadata

### 4.1 Manifest

manifest 使用 RFC 8785 JSON Canonicalization Scheme 产生的 canonical UTF-8
JSON。原因是 manifest 主要用于检查、审计、转换和人工诊断，JSON 的可见性
优于二进制对象格式；高频 tensor 定位由 binary index 完成，不再依赖 JSON
扫描性能。

manifest MUST：

- 拒绝重复 key；
- 拒绝非 UTF-8；
- 限制最大字节数、最大深度和最大字符串长度；
- 使用确定性 key 顺序和数字表示；
- 记录 `schema_version` 与 `model_id`；
- 记录 architecture、source revision、tokenizer、compatibility 和 tensor count；
- 记录每个 tensor 的 `tensor_id`、name、logical shape、logical dtype 和 semantic role。

manifest MUST NOT 保存任意可执行对象。

### 4.2 Metadata

metadata 使用 canonical CBOR map，采用 definite-length 编码。CBOR 只用于结构化 metadata，不用于大 tensor payload。

允许的值类型：

- UTF-8 string；
- unsigned/signed integer；
- finite float32/float64；
- boolean；
- bounded array；
- bounded map。

metadata MUST 设置最大深度、最大 map/array 项数和最大字符串字节数。metadata 中的 model/runtime hint 只能影响优化，不能改变 tensor 的正确解释。

## 5. Binary index

index 由三个确定性区域组成：

```text
string table
tensor records
chunk records
```

### 5.1 Tensor record

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
stride_reference (optional)
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

rank、shape 和 stride 使用 64 位无符号整数。实现 MUST 限制最大 rank，防止恶意 shape 造成分配或乘法溢出。

v2.0 内建 layout：

```text
row_major
column_major
blocked
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

无压缩 chunk MUST 可以通过 offset/length 直接 mmap。压缩 chunk MUST 可以独立解压和独立校验，禁止使用跨 chunk 的连续压缩流。

### 6.2 Codec

v2.0 core codec：

```text
0 = none       (MUST)
1 = zstd       (SHOULD)
2 = lz4        (MAY)
```

默认 encoder SHOULD 对推理热路径使用 `none`，对冷数据、远端对象存储或低带宽传输使用 zstd/lz4。自定义 codec 不属于 v2.0 core，必须通过明确的 extension ID 和能力协商。

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

v2.0 SHOULD 内建 F32、F16、BF16、FP8、INT8、INT4、NF4 和当前 QuantInfer 使用的 Q4_0/Q4_K/Q5_K/Q6_K/Q8_0。具体内建集合必须通过 golden files 固化，不得只写在文档中。

## 7. 完整性与安全

### 7.1 Hash

SHA-256 是 v2.0 必选算法，以保持跨语言和现有 QuantInfer 生态的可用性。BLAKE3 可以作为后续可选算法，但不能成为 v2.0 的唯一依赖。

hash 层级为：

```text
chunk hash
tensor hash = canonical tensor record + chunk hashes
root hash = canonical manifest + metadata + index + tensor/chunk digest list
```

`verify --fast` 至少验证 header、manifest、metadata、index 和 root hash。`verify --full` 必须读取并验证所有 payload chunk。

### 7.2 Signature

签名 envelope 在 v2.0 中只定义位置和长度，不强制具体签名算法。Ed25519/COSE 属于 v2.1 候选，避免在 v2.0 同时锁定密钥分发、证书链和签名格式。

### 7.3 禁止项

QIF v2 MUST NOT：

- 执行反序列化代码；
- 保存 pickle 或任意 Python object；
- 允许 metadata 路径穿越；
- 允许 unchecked allocation；
- 允许 NaN/Infinity 进入标准浮点 payload；
- 允许未知 mandatory encoding 被当作其他 dtype 读取。

## 8. QIF2-PACKAGE

QIF2-PACKAGE 由一个 package manifest 和多个 shard 构成：

```text
qif.json
index.qifi
shard-00000.qifd
shard-00001.qifd
...
checksums.sha256
```

package manifest MUST 为每个 shard 记录：

- 相对路径或受限 URI；
- 文件大小；
- shard hash；
- global root hash；
- tensor 的 global shape；
- logical offset/length；
- mesh axis、rank 和 shard coordinate。

路径 MUST 禁止 `..`、绝对路径和未授权 scheme。package 的发布必须使用临时目录、完整校验和原子 rename；未完成 package 不得被 reader 视为有效模型。

## 9. Runtime API

`quantinfer-format` 应新增独立的 file-backed reader，而不是继续把 QIF 伪装成 `Vec<u8>`：

```rust
QifFileReader::open(path)
QifFileReader::manifest()
QifFileReader::metadata()
QifFileReader::tensor_info(name)
QifFileReader::tensor_region(name)
QifFileReader::read_range(name, range)
QifFileReader::mmap_range(name)
QifFileReader::verify(mode)
```

API 要求：

- 打开 index 不读取全部 payload；
- uncompressed chunk 可以零拷贝映射；
- compressed chunk 只分配当前请求所需内存；
- reader 可选择 eager、lazy、parallel verification；
- 文件、内存 buffer 和 package shard 使用同一逻辑接口；
- Rust/C/C++ binding 不依赖 PyTorch、Python、ONNX Runtime 或 llama.cpp。

## 10. 性能目标与验收边界

以下是必须验证的工程目标，不是未经测试的既定优势：

1. index open 不读取 payload；
2. 只读取一个 tensor 时，不触碰其他 tensor 的 payload；
3. 无压缩路径能返回 mmap/range view；
4. 压缩路径不会把整个模型解压到内存；
5. QIF2-SINGLE 与 QIF2-PACKAGE 对相同 tensor 产生一致逻辑值；
6. 只改变 shard/rank 布局时无需重新编码 tensor；
7. root/full verification 在损坏、截断、重叠、越界和错误 codec 上稳定失败；
8. Rust、C 和 Python golden readers 对同一 fixtures 得出一致 tensor info 和 bytes；
9. 与 SafeTensors/GGUF 的 benchmark 必须报告：open latency、first-tensor latency、RSS、page faults、selected-layer read bandwidth、full-load time 和验证吞吐；
10. 不得把“比 SafeTensors 快多少”写入规范，除非由可复现基准证明。

## 11. 从当前仓库迁移

迁移顺序：

1. 新增 `qif-v2` module，禁止在 v2 module 复用 v1 header parser。
2. 将当前 `qif` module 标记为 legacy-only，停止生成 v1 artifact。
3. 增加 QIF2 binary index、file-backed reader、single-file writer。
4. 增加 QIF2 package/shard writer 和 atomic publish。
5. 增加 QIF2 golden fixtures、malformed fixtures、fuzz target、C/Python smoke tests。
6. 新 CLI 默认输出 QIF2；旧 v1 只能通过显式 `--legacy-qif-v1` 读取。
7. 完成 GGUF/SafeTensors → QIF2 转换后，再删除 v1 writer 和 v1 默认 reader。
8. 更新 `docs/architecture/design-decisions.md`、`docs/architecture/mvp-qwen3.md`、`docs/rfcs/README.md`，明确 v1 已废弃、v2 是唯一新格式。

## 12. 设计结论

QIF v2 的实际价值不是声称全面击败 SafeTensors、GGUF 或 DCP，而是明确解决它们之间的组合空白：

- 继承 SafeTensors 的无代码执行和简单 tensor 读取边界；
- 继承 GGUF 的 self-describing metadata、量化和单文件推理思路；
- 吸收 DCP/Orbax/Universal Checkpointing 的 global shape、offset、shard 和 resharding 元数据；
- 吸收 fastsafetensors、DataStates-LLM、TierCheck 的批量、异步、分层和按需 I/O 思路；
- 保持 QuantInfer 自有的 logical descriptor、semantic tensor role 和 backend capability 边界。

因此，QIF v2.0 的成功标准是：**同一套可验证权重语义，既能在一个文件中低依赖 mmap 加载，也能在多个 shard、不同设备布局和远端范围读取中保持一致解释。**

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
