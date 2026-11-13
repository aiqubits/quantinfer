# QI-004 设计

F32 存储从 `Vec<f32>` 改成 `Arc<Vec<f32>>`。U16 和 U8 仍是 `Vec`。`len` 通过 `Deref` 保持原样。

`as_f32_mut` 不能把 `Arc::make_mut` 的结果直接当返回值。Rust 1.99 会把它推成 `Arc<[f32]>`。实现先把结果标成 `&mut Vec<f32>`，再取 `as_mut_slice`。写一条视图会触发 copy-on-write，源张量保持原值。

`from_f32` 和标量构造把数据放进 `Arc`。`reshape` 先克隆 `Arc`。F32 且 stride 已经是行主序时，克隆只增加引用计数。stride 不是行主序时返回 `CoreError::Unsupported`，信息是 `reshape requires a contiguous row-major tensor`。

`Tensor::view` 造出 `TensorView`。它不再是 `Copy`，因为自己持有 `Shape` 和 `Strides`。方法都拿 `&self`，调用处不用改。字段是 `parent`、`offset`、`shape`、`strides`。

视图操作：

- `as_f32` 只接受连续切片。不连续时返回 `tensor view is not a contiguous slice`。范围用 `checked_add`。
- `get_f32` 按 stride 读一个元素，越界和秩不匹配返回错误。
- `row_slice(start, end)` 用于秩 2，按第 0 维的 stride 算 offset。
- `transpose` 交换维和 stride，只接受秩 2。
- `narrow(element_offset, shape)` 只接受连续且范围合法的视图，新形状用行主序 stride。
- `is_contiguous` 从最后一维往前走，跳过长度不大于 1 的维，要求 stride 等于期望值，然后期望值乘上这一维。

量化张量没有改。qwen3、llama 和 imatrix 里的 `reshape` 调用保持原样，测试仍然通过。
