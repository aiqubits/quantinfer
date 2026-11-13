# QI-004 带 offset 的 TensorView

## 目标

让 `TensorView` 能表示带 offset 和 stride 的子视图，包括行切片和转置。行主序的 `reshape` 在可以共享存储时不再复制 F32 数据。后面的零拷贝加载和 attention 融合用这个抽象。

## 约束

- `quantinfer-core` 保持 `forbid(unsafe_code)`。视图用生命周期借用。
- 旧的 `Tensor` 调用方式保持可用。当时没有别的 crate 直接构造 `TensorView`。
- 越界返回 `CoreError`，不 panic。

## 上下文

`TensorView` 原来是整张张量的包装，`reshape` 会克隆存储。Gitee 上的观察记录从 v0.2 起就点过这个缺口。

## 关联

建议排在 QI-001 和 QI-003 之后。它是 QI-005 的前置，也是 QI-006 和 QI-010 的软前置。

## 非目标

- 不做通用的任意 stride 迭代器。
- 不改 `QuantizedTensor` 系列的存储。
- 不做设备端视图。

## 验收

- 行区间、转置、带 offset 的连续子视图都能表达。连续视图的 `as_f32` 返回借用切片。
- 行主序 `reshape` 共享存储，用指针相等证明。
- 越界和形状不匹配有单元测试。
- `cargo test --workspace --locked` 通过，已有数值门禁不漂移。
