// Tile size and argument types are shared with the host launch code.
constexpr unsigned int tile = 16;

__device__ float read_weight(const unsigned int *weights, unsigned int index,
                             unsigned int encoding) {
    if (encoding == 0) {
        return __uint_as_float(weights[index]);
    }
    const unsigned int lane = index % 32;
    if (encoding == 1) {
        const unsigned int block = (index / 32) * 5;
        const unsigned int word = weights[block + 1 + lane / 8];
        const unsigned int code = (word >> ((lane % 8) * 4)) & 15;
        return __uint_as_float(weights[block]) * (static_cast<float>(code) - 8.0f);
    }
    const unsigned int block = (index / 32) * 9;
    const unsigned int word = weights[block + 1 + lane / 4];
    const unsigned int code = (word >> ((lane % 4) * 8)) & 255;
    const int signed_code = static_cast<int>(code ^ 128) - 128;
    return __uint_as_float(weights[block]) * static_cast<float>(signed_code);
}

extern "C" __global__ void quantinfer_matmul(
    const float *left, const unsigned int *right, float *output,
    unsigned int rows, unsigned int inner, unsigned int columns,
    unsigned int encoding, unsigned int output_rows, unsigned int column_tiles) {
    __shared__ float left_tile[tile][tile];
    __shared__ float right_tile[tile][tile];
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int row = (blockIdx.x / column_tiles) * tile + y;
    const unsigned int column = (blockIdx.x % column_tiles) * tile + x;
    float sum = 0.0f;
    for (unsigned int base = 0; base < inner;) {
        const unsigned int lanes = min(tile, inner - base);
        left_tile[y][x] = row < rows && x < lanes ? left[row * inner + base + x] : 0.0f;
        float weight = 0.0f;
        if (column < columns && y < lanes) {
            const unsigned int channel = base + y;
            const unsigned int index = output_rows != 0
                ? column * inner + channel : channel * columns + column;
            weight = read_weight(right, index, encoding);
        }
        right_tile[y][x] = weight;
        __syncthreads();
        for (unsigned int lane = 0; lane < lanes; ++lane) {
            sum += left_tile[y][lane] * right_tile[lane][x];
        }
        __syncthreads();
        base += lanes;
    }
    if (row < rows && column < columns) {
        output[row * columns + column] = sum;
    }
}
