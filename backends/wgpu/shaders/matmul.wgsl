struct Matrix {
    rows: u32,
    inner: u32,
    columns: u32,
    encoding: u32,
    output_rows: u32,
}

@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> matrix: Matrix;

var<workgroup> left_tile: array<array<f32, 8>, 8>;
var<workgroup> right_tile: array<array<f32, 8>, 8>;

fn weight_at(inner: u32, column: u32) -> f32 {
    let index = select(
        inner * matrix.columns + column,
        column * matrix.inner + inner,
        matrix.output_rows != 0u,
    );
    if matrix.encoding == 0u {
        return bitcast<f32>(weights[index]);
    }
    let lane = index % 32u;
    if matrix.encoding == 1u {
        let start = (index / 32u) * 5u;
        let word = weights[start + 1u + lane / 8u];
        let code = (word >> ((lane % 8u) * 4u)) & 15u;
        return bitcast<f32>(weights[start]) * (f32(code) - 8.0);
    }
    let start = (index / 32u) * 9u;
    let word = weights[start + 1u + lane / 4u];
    let code = (word >> ((lane % 4u) * 8u)) & 255u;
    let signed = select(i32(code), i32(code) - 256, code >= 128u);
    return bitcast<f32>(weights[start]) * f32(signed);
}

@compute @workgroup_size(8, 8)
fn matmul(@builtin(global_invocation_id) global: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    var sum = 0.0;
    for (var base = 0u; base < matrix.inner; base += min(8u, matrix.inner - base)) {
        let lanes = min(8u, matrix.inner - base);
        left_tile[local.y][local.x] = 0.0;
        right_tile[local.y][local.x] = 0.0;
        if global.y < matrix.rows && local.x < lanes {
            left_tile[local.y][local.x] = activations[global.y * matrix.inner + base + local.x];
        }
        if global.x < matrix.columns && local.y < lanes {
            right_tile[local.y][local.x] = weight_at(base + local.y, global.x);
        }
        workgroupBarrier();
        for (var inner = 0u; inner < lanes; inner += 1u) {
            sum += left_tile[local.y][inner] * right_tile[inner][local.x];
        }
        workgroupBarrier();
    }
    if global.y < matrix.rows && global.x < matrix.columns {
        output[global.y * matrix.columns + global.x] = sum;
    }
}
