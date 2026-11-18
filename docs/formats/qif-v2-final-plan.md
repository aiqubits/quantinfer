# QIF v2.0 权重容器标准方案

对外 artifact、CLI 和 Rust API 统一使用 **QIF** 名称；QIF2-SINGLE、QIF2-PACKAGE
及 QIF v1 只用于内部 wire/version 实现与历史文档，不作为对外能力标识。

状态：QIF-SINGLE、QIF-PACKAGE、独立压缩 chunk 和 QIF-SINGLE 的 Ed25519 签名 envelope
已有实现，本文固定 wire 字段；跨语言 golden reader、带 ASan/覆盖率的 seeded fuzz smoke test 和
benchmark 已接入 CI。功能存在与 CI 通过不等于全部验收完成：跨语言验证范围、
性能测量口径、默认模型加载器接入和长时 fuzzing 仍须分别验证。

QIF v1 已停止作为 QuantInfer 的目标格式。当前仓库中的 QIF v1 reader、writer 和
历史测试只保留为旧 artifact 的读取与迁移参考；新 GGUF 转换器写入当前 QIF（内部 wire version 为 QIF2-SINGLE），
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

QIF v2.0 冻结两种容器 profile；对外 profile 名称统一为 QIF，内部实现仍以
QIF2 wire version 区分：

- **QIF-SINGLE**（内部 QIF2-SINGLE）：一个文件，适合本地、边缘和单机服务。
- **QIF-PACKAGE**（内部 QIF2-PACKAGE）：`qif.json`、`index.qifi`、`metadata.qmb` 和多个 `.qifd` shard，适合对象存储、大模型和并行读取。

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

格式库允许检查和转存带有未来运行时要求的 artifact；执行模型前必须单独检查
compatibility。`minimum_quantinfer_version` 指 QuantInfer runtime 的 SemVer，
不是 QIF wire version。当前 runtime 检查最低版本、已登记的 operation 名称与
所选 provider 的能力；数值 kernel requirement 指定时必须大于零。
`required_features` 和 tensor 的 `backend_requirements` 尚无已实现的字符串
registry，非空必需列表会被明确拒绝，不会按可忽略 hint 处理。
operation 名称固定为 `matmul`、`quantized_matmul`、`attention`、`rms_norm`、
`activation`、`rope`、`embedding`；未知名称拒绝执行。版本、feature 和 provider
检查不能替代格式完整性验证。

### 4.2 Metadata

metadata 使用 QuantInfer 定义的 QMB（QIF Metadata Binary）typed map，而不是
强制引入第三方 object serialization。QMB 的 key 按 UTF-8 编码字节排序，所有
map/array 使用 definite-length 编码，因此同一逻辑 metadata 只有一种 wire
表示。`quantinfer-cli inspect --input MODEL.qif|PACKAGE_DIR --json` 按原生控制区
展示完整 metadata、manifest、logical shape、encoding/codec ID 与 alias。
metadata 的 JSON 展示采用 `{ "type": "U32", "value": 1 }` 等带类型表示，保留
QMB 数值宽度、数组与 Map；该表示是检查结果，不是另一种权重 wire 编码。
兼容命令 `inspect-gguf` 保留 GGUF 视图，省略含 Map 的 metadata 条目。
完整原生 metadata 也可由 `qif::Reader::metadata()` 或
`PackedTensorReader::qif_metadata()` 取得。
JSON 不是 payload wire format。

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

metadata MUST 设置最大深度、最大 map/array 项数和最大字符串字节数。metadata 中的
model/runtime hint 只能影响优化，不能改变 tensor 的正确解释。保留键
`quantinfer.qif.ggml_dimension_order` 是兼容 GGUF 转换的结构字段，必须为 bool，
控制兼容视图的维度顺序，不属于可忽略的 hint。包含 Map 的 metadata 保留在原生
视图中，不强制转换为 GGUF；兼容视图省略这些条目。

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

rank 使用 `u32`，各 dimension 使用 `u64`；rank 存于 shape table，不是 tensor
record 的独立字段。stride 在 v2.0 不存储，后续 layout profile
不得复用当前 tensor record 的 reserved 字段。实现 MUST 限制最大 rank，防止恶意
shape 造成分配或乘法溢出。

QIF2-SINGLE v2.0 内建 layout：

```text
row_major
ggml_output_rows
```

`ggml_output_rows` 是 QIF2 的规范字符串；QuantInfer 内部旧的
`output_rows` descriptor 会在 QIF2 writer 入口 canonicalize 为该名称。

新的 layout 必须使用新的 registry ID；不能通过复用旧 ID 改变含义。

### 5.3 Alias

如果多个逻辑 tensor 共享相同物理 payload，除第一个 tensor 外，其余 tensor MUST 使用 `alias_target`，不得重复写入相同 bytes。

当前 QIF2-SINGLE 与 QIF2-PACKAGE writer 会在 tensor descriptor 和原始 payload
一致时自动去重并生成 alias record；reader、file-backed reader 和 package reader
都会沿 alias 解析到物理 tensor，并继续执行相同的 checksum 校验。

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

v2.0 的 section 和 chunk offset MUST 是 4096 字节的倍数。默认 alignment 为
4096 字节；实现 MAY 使用更大的 64 KiB 或 2 MiB 对齐。选择的 alignment 必须
是 2 的幂，且不小于 4096 字节。

无压缩 chunk MUST 可以通过 offset/length 直接读取；压缩 chunk MUST 可以独立解压和
独立校验，禁止使用跨 chunk 的连续压缩流。mmap 的零拷贝 tensor 视图只适用于
无压缩 chunk；映射压缩文件仍需要为解压后的数据分配空间。
物理 chunk 的 `raw_length` 和 `stored_length` MUST 大于零；空 chunk 不允许出现在
index 中，避免不同读取与验证路径对空压缩流产生不同解释。

### 6.2 Codec

QIF2-SINGLE/QIF2-PACKAGE v2.0 core codec：

```text
0 = none       (MUST)
1 = zstd       (supported)
2 = lz4        (supported)
```

每个压缩 chunk 独立编码、独立解码和独立校验；reader 不使用跨 chunk 连续压缩流。
内部 `Reader::tensor_data` 与 file/package reader 会按需解压当前 tensor 数据。
解码输出 MUST 受 index 的 `raw_length` 约束，禁止先无限展开再检查长度。
当前实现的压缩资源 profile 限制单 chunk 的 `raw_length` 不超过 128 MiB、
`stored_length` 不超过 129 MiB，并限制 zstd 解码 window 不超过 128 MiB；
超限输入在分配压缩块或解码输出之前返回错误。这是当前 reader 的资源限制，
不改变 wire 中的长度字段。writer 默认将压缩 tensor 切分为不超过 4 MiB 的
raw chunk；PACKAGE 还受 shard 大小与压缩后大小约束。无压缩 chunk 不受上述
压缩块大小上限限制。LZ4 的长度前缀 MUST 与 `raw_length` 相同，并在输出分配前检查。

文件和 PACKAGE 的 `verify()` 增量计算 tensor hash 并校验有限数值：无压缩块
使用固定大小缓冲区，压缩块逐块解码，alias 不重复物化或校验相同物理 payload。
这不表示任意资源配置都能加载整个模型；文件后端请求 tensor 时仍需物化所选
tensor 的 raw bytes，Dense model 构造还需要解码后的权重。
带内存预算的默认模型入口会在解压前检查最低存储需求：原生 QIF 文件后端检查
序列化控制区加物理权重大小，owned buffer 入口还计入其保留的输入。alias
只计一次；该检查不包括 allocator、解码 workspace 或推理状态的全部峰值开销。

packed reader 原生保留 QIF Map 元数据；其 GGUF 兼容 metadata 视图只暴露可转换
的值。模型适配器通过 `metadata_value()` 查询已知参数，遇到包含 Map 的值时
报类型错误，不能将“已存在但类型不支持”当作缺失并使用默认值。未知可选 hint
无需被模型适配器读取，仍可被忽略。

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

上述固定块编码要求 `group_size=None`、`scale_dtype=F16`、
`zero_point_dtype=None`；Q8_0/Q4_0 的 block size 为 32，K-quant 为 256。
packing 分别为 `scalar`、`nibble_split_half`、`k_quant`。这些字段与物理 encoding
MUST 一致；其它 packing、scale、axis 或 group 表达不能复用现有 encoding ID。
量化 axis 支持整个 tensor 的 `Tensor`；兼容转换产生的二维
`ggml_output_rows` descriptor 还支持 `Axis(0)`。其它 axis 不在当前编码集合内。
浮点 encoding 使用 `packing=none`，且不得声明量化 block/group/scale/zero point。

## 7. 完整性与安全

### 7.1 Hash

SHA-256 是 v2.0 必选算法，以保持跨语言和现有 QuantInfer 生态的可用性。BLAKE3 可以作为后续可选算法，但不能成为 v2.0 的唯一依赖。

v2.0 的 hash 层级为：

```text
chunk hash = SHA-256(stored chunk bytes)
tensor hash = SHA-256(raw tensor bytes)
root hash = SHA-256(canonical manifest + canonical metadata + binary index)
```

复用 `ModelManifest` 的 tensor checksum 按 SHA-256 十六进制值比较，允许大写
或小写十六进制字母；writer 输出小写。root hash 仍覆盖实际 canonical manifest
字节，不按 checksum 字符串大小写进行额外规范化。

reader 打开时至少验证 header、manifest、metadata、index 和 root hash；显式
`verify()` 必须读取并验证所有 tensor、payload chunk，并拒绝非有限标准浮点/量化 scale。
v2.0 不定义命令行
`verify --fast/--full` 选项。

### 7.2 Signature

QIF-SINGLE v2.0 定义固定 108 字节 Ed25519 envelope：`QSIG` magic、版本、算法 ID、公钥和
签名。签名覆盖 root hash；reader 验证 envelope，但信任根和密钥分发由发布系统负责。
QIF-PACKAGE 当前使用 package root hash、metadata/index hash 和 shard hash；package
级签名 envelope 不属于当前 v2.0 wire core。
COSE、证书链和密钥轮换不属于 v2.0 wire core。

### 7.3 禁止项

QIF v2 MUST NOT：

- 执行反序列化代码；
- 保存 pickle 或任意 Python object；
- 允许 metadata 路径穿越；
- 允许 unchecked allocation；
- 允许 NaN/Infinity 进入标准浮点 payload；
- 允许未知 mandatory encoding 被当作其他 dtype 读取。

## 8. QIF-PACKAGE

QIF-PACKAGE 由一个 package manifest、一个全局 binary index 和多个独立 shard 构成。
其 `qif.json` 的 `profile` 固定为 `QIF-PACKAGE`；reader 也接受历史生成器写出的
`QIF2-PACKAGE` profile，并在其它内容校验通过后按同一 wire schema 读取：

```text
qif.json
metadata.qmb
index.qifi
shard-00000.qifd
shard-00001.qifd
...
```

package manifest 为每个 shard 记录：

- 相对路径；受限 URI 作为后续 transport profile 保留，v2.0 当前 reader 只接受包根目录内的本地相对路径；
- 文件大小；
- shard hash；
- mesh axis、rank 和 shard coordinate；
- 所属 tensor 名称；
- package root hash（覆盖 root_hash 置空后的 canonical package manifest、metadata、global index 和 shard hashes）；
- model manifest 中的 global shape 与 tensor checksum。global index 的 chunk record
  使用 tensor-relative `raw_offset/raw_length` 和 shard-relative
  `stored_offset/stored_length`，因此一个 tensor 可以跨多个 shard 重组。

路径 MUST 禁止 `..`、绝对路径和未授权 scheme。package 的发布必须使用临时目录、完整校验和原子 rename；未完成 package 不得被 reader 视为有效模型。

writer 的 `max_shard_bytes` 限制最终 shard 文件大小，包含 alignment padding 和
codec framing。writer 必须在放置前计算对齐后的终点；压缩后超过限制时缩小 raw
chunk，若连一个字节的 codec frame 也无法容纳则返回错误。

## 9. Runtime API

`quantinfer-format::qif` 的内存、file-backed、mmap 和 package reader 是对外稳定 API；
wire-version implementation remains internal. 当前对外 API 为：

```rust
quantinfer_format::qif::Reader::open(bytes)
quantinfer_format::qif::Reader::manifest()
quantinfer_format::qif::Reader::metadata()
quantinfer_format::qif::Reader::tensor_info(name)
quantinfer_format::qif::Reader::tensor_bytes(name)
quantinfer_format::qif::Reader::read_range(name, range)
quantinfer_format::qif::Reader::verify()
quantinfer_format::qif::FileReader::open(path)
quantinfer_format::qif::FileReader::read_range(name, range)
quantinfer_format::qif::FileReader::verify()
unsafe { quantinfer_format::qif::MappedFile::open(path) }
quantinfer_format::qif::MappedFile::reader()
quantinfer_format::qif::PackageReader::open(path)
quantinfer_format::qif::PackageReader::read_range(name, range)
quantinfer_format::qif::PackageReader::verify()
quantinfer_format::qif::write_qif_with_codec(
    ..., quantinfer_format::qif::Codec::Zstd /* or Codec::Lz4 */)
quantinfer_format::qif::write_qif_signed(
    ..., quantinfer_format::qif::Codec::None, signing_key)
```

API 要求：

- 内存 reader 打开时验证 header、manifest、metadata、index 和 root hash；
- `read_range` 只复制请求的 tensor 子范围；
- QIF-SINGLE/PACKAGE 提供按需 file-backed range read，并提供只读 `qif::MappedFile` 映射入口；
  mmap 的 unsafe 操作被隔离在单独模块。`MappedFile::open` 是 unsafe API，调用者
  必须保证映射及所有借用 view 存活期间，底层文件不被任何线程或进程修改/截断，
  并仍需通过同一 QIF reader 校验。只读映射和 hash 校验不提供文件不变性保证；
  不能满足该前提时使用安全的 `FileReader`；
- reader 可选择调用 `verify` 执行全 payload 校验；
- CLI `inspect` 仅校验控制区并标记 `payload_verified=false`；`verify` 对原生
  SINGLE/PACKAGE 执行流式全量校验，失败返回非零退出码。嵌入签名的数学有效性
  不等于发布者已受信任，检查结果的 `publisher_trust=not_evaluated` 明确此边界；
- 文件、内存 buffer 和 package shard 使用同一逻辑 tensor/chunk 校验规则；
- Rust/C/C++ binding 不依赖 PyTorch、Python、ONNX Runtime 或 llama.cpp。

默认 `ModelSource` 现在对原生 QIF 单文件与 `qif.json` PACKAGE 使用文件后端：
open 只读取控制区，typed tensor decoder 按需读取所选 tensor、检查 raw hash
与有限值，转换完成即释放原始字节缓冲区。显式 `tensor_bytes()` 借用接口按需
缓存，alias 与 reader clone 共享同一物理缓存；`verify_qif()` 全量流式校验不填充缓存。
无 unsafe mmap 前提的路径使用普通文件 I/O。

GGUF 继续使用 owned reader；`manifest.json` 模型包保留其先验证完整文件快照的
路径。Dense runtime 构造时仍会将模型需要的权重解码为 Core tensor；文件后端
减少原始权重副本，不等于已实现按执行层换入换出的 out-of-core 推理。

## 10. 性能目标与验收边界

以下是必须验证的工程目标，不是未经测试的既定优势：

1. file-backed index open 不读取 payload（当前 `qif::FileReader::open` 已满足）；
2. 只读取一个 tensor 时，不触碰其他 tensor 的 payload（当前 file-backed `read_range` 已满足）；
3. 无压缩路径能返回 mmap/range view（当前 `qif::MappedFile` 与 file-backed range read 已满足）；
4. 压缩路径只解压当前请求的 tensor/chunk（当前 codec reader 已满足）；
5. QIF2-SINGLE 与 QIF2-PACKAGE 对相同 tensor 产生一致逻辑值（当前 package tests 已覆盖）；
6. package 通过全局 index 改变 shard 分配而不改变 tensor 编码（当前已有确定性的 `plan_qif_v2_reshard` API、跨 shard 规划和测试；writer 仍会结合 codec 大小与对齐执行最终物理放置）；
7. root/full verification 在损坏、截断、重叠、越界和错误 codec 上稳定失败；
8. Rust reader 与独立 C/Python golden readers 对固定 unsigned/uncompressed F32 fixture 得出一致 header、index 字段、section/checksum 和 payload bytes；CI 比较原始及控制区重排两个布局，并要求三者拒绝四类损坏输入。C/Python 工具限于此 golden wire profile，不是通用 ModelManifest/QMB、codec、PACKAGE 或签名验证器；范围见 [QIF conformance](qif-conformance.md)；
9. `cargo run -p quantinfer-xtask -- qif-bench --qif PATH --safetensors PATH --gguf PATH --tensor NAME` 在独立进程中报告三种格式的 open、selected-tensor range、open-to-first-range、artifact size、RSS/high-water 和 page-fault 指标；GGUF open 包含整文件读入，QIF full-verify 的耗时与资源统计单列。输出说明 eager/file-backed 加载方式、dtype、范围字节数/hash 和未控制的缓存状态。不同名称可用 `--qif-tensor`、`--safetensors-tensor`、`--gguf-tensor` 显式对应；测量范围和比较限制见 [benchmarks](../../benches/README.md)；
10. 不得把“比 SafeTensors 快多少”写入规范，除非由可复现基准证明。

## 11. 从当前仓库迁移

迁移顺序：

1. 新增 `qif-v2` module，禁止在 v2 module 复用 v1 header parser（已完成）。
2. 将当前 legacy `qif` module 标记为 legacy-only，新的 GGUF 转换器写入当前 QIF（内部 wire version 为 QIF2，已完成）。
3. 增加 QIF2 binary index、内存 reader、single-file writer（已完成）。
4. file-backed reader、mmap/range view、QIF2 package/shard writer 已完成。
5. QIF2 的 Rust fixture、dependency-free Python reader、独立 C wire golden reader 和
   `qif_v2_reader`、`qif_v2_package` fuzz target 已接入。CI 使用固定 nightly 和
   cargo-fuzz，检查非零覆盖率；内存/文件/mmap、codec/full verify 和 PACKAGE index
   都有受限运行路径。输入、输出和时长限制及未覆盖的 package JSON/path/shard
   边界见 [fuzz/README.md](../../fuzz/README.md)，不能将 smoke 解释为长时或完整 fuzz 验收。
6. 新 CLI 默认输出 QIF；默认 runtime 拒绝旧 v1，旧 v1 module 仅保留在格式 crate 内供历史 fixtures 使用。
7. GGUF → QIF 已由 `from-gguf` 实现；SafeTensors 转换器通过
   `--output-format qif` 发布当前 QIF。二者使用 `--qif-profile single|package`、
   `--qif-codec none|zstd|lz4`，PACKAGE 还接受 `--max-shard-bytes N`。内部仍复用
   已验证的 GGUF 语义映射；发布器的转换/写入阶段不是全模型内存上界保证。
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
