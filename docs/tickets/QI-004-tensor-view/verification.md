# QI-004 验证

在 `ee63330` 上跑过：

```sh
cargo test --workspace --locked
cargo clippy --locked -p quantinfer-core --all-targets -- -D warnings
```

两次都通过。`numerical_gates` 在工作区测试里通过，没有单独改容差。

覆盖的单测：

- `reshape` 之后 `as_ptr` 相等。
- 对 reshape 结果做可变借用会 copy-on-write，源数据保持 `[1.0, 2.0]`。
- 3×2 张量上，`row_slice(1, 3)` 的 offset 是 2，值是 `[3, 4, 5, 6]`。
- `narrow(2, [2, 2])` 的指针等于原张量从下标 2 开始的切片。
- 转置后形状是 `[2, 3]`，`as_f32` 失败，`get_f32([1, 2])` 是 6，`get_f32([0, 1])` 是 3。
- 越界、错误秩、对转置结果再 `narrow` 都返回错误。
- 非秩 2 的转置被拒绝。

当时仓库里没有别的 `TensorView` 调用点，去掉 `Copy` 没有打破编译。
