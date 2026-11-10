use std::borrow::Cow;
use std::collections::BTreeSet;
use std::sync::mpsc;
use std::time::Duration;

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{
    DType, Device, QuantizationScheme, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K,
    QuantizedTensorQ5K, QuantizedTensorQ6K, Shape, Tensor, GGML_OUTPUT_ROWS_LAYOUT,
};
use quantinfer_kernel::{KernelBackend, KernelError, Result};
use quantinfer_runtime::{KernelCapability, Operation, RuntimeProvider};

const TILE: u32 = 8;
const GPU_TIMEOUT: Duration = Duration::from_secs(30);

/// F32, Q4_0, and Q8_0 projections on a native wgpu adapter.
#[derive(Debug, Clone)]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_info: wgpu::AdapterInfo,
    capabilities: Vec<KernelCapability>,
    cpu: CpuScalar,
}

impl WgpuBackend {
    /// Request a native adapter. Enable the fallback adapter for software Vulkan tests.
    pub fn open(force_fallback_adapter: bool) -> Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(unavailable(
                "packed shader storage requires a little-endian host",
            ));
        }
        if wgpu::Instance::enabled_backend_features().is_empty() {
            return Err(unavailable("this target has no compiled native adapter"));
        }
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(unavailable)?;
        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("QuantInfer"),
            required_limits,
            ..Default::default()
        }))
        .map_err(unavailable)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("QuantInfer tiled matmul"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matmul.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("QuantInfer matmul"),
            layout: None,
            module: &shader,
            entry_point: Some("matmul"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(unavailable(error));
        }
        Ok(Self {
            device,
            queue,
            pipeline,
            adapter_info,
            capabilities: capabilities(),
            cpu: CpuScalar::new(),
        })
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    fn multiply(&self, lhs: &Tensor, rhs: Weights<'_>, output_rows: bool) -> Result<Tensor> {
        let &[rows, inner] = lhs.shape().dims() else {
            return Err(invalid("left matrix must have rank 2"));
        };
        let &[rhs_inner, columns] = rhs.shape().dims() else {
            return Err(invalid("right matrix must have rank 2"));
        };
        if rows == 0 || inner == 0 || columns == 0 || inner != rhs_inner {
            return Err(invalid(
                "matrix dimensions must be nonzero and share an inner dimension",
            ));
        }
        if output_rows && !inner.is_multiple_of(32) {
            return Err(invalid(
                "packed output rows require an input width divisible by 32",
            ));
        }
        let matrix = [
            shader_size(rows)?,
            shader_size(inner)?,
            shader_size(columns)?,
            rhs.encoding(),
            u32::from(output_rows),
        ];
        let groups = (matrix[2].div_ceil(TILE), matrix[0].div_ceil(TILE));
        let limits = self.device.limits();
        if groups.0.max(groups.1) > limits.max_compute_workgroups_per_dimension {
            return Err(unavailable("matrix exceeds the adapter dispatch limit"));
        }
        shader_elements(matrix[0], matrix[1])?;
        shader_elements(matrix[1], matrix[2])?;
        let result_bytes = u64::from(shader_elements(matrix[0], matrix[2])?) * 4;
        let lhs = lhs.as_f32()?;
        let limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if (lhs.len() as u64) * 4 > limit || rhs.byte_len()? > limit || result_bytes > limit {
            return Err(unavailable(format!(
                "matrix exceeds the adapter's {limit}-byte storage limit"
            )));
        }
        let weights = rhs.bytes()?;
        self.dispatch(
            &matrix,
            groups,
            bytemuck::cast_slice(lhs),
            &weights,
            result_bytes,
        )
    }

    fn dispatch(
        &self,
        matrix: &[u32; 5],
        groups: (u32, u32),
        lhs: &[u8],
        rhs: &[u8],
        result_bytes: u64,
    ) -> Result<Tensor> {
        let memory_scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let upload = |label, contents: &[u8], usage| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: contents.len() as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, contents);
            buffer
        };
        let left = upload("activations", lhs, wgpu::BufferUsages::STORAGE);
        let right = upload("weights", rhs, wgpu::BufferUsages::STORAGE);
        let shape = upload(
            "matrix",
            bytemuck::cast_slice(matrix),
            wgpu::BufferUsages::UNIFORM,
        );
        let result_buffer = |label, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: result_bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let output = result_buffer(
            "output",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = result_buffer(
            "readback",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let bindings = [&left, &right, &output, &shape];
        let entries = std::array::from_fn::<_, 4, _>(|index| wgpu::BindGroupEntry {
            binding: index as u32,
            resource: bindings[index].as_entire_binding(),
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul buffers"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, result_bytes);
        let (sender, receiver) = mpsc::channel();
        let submission = self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let poll = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(GPU_TIMEOUT),
        });
        let validation_error = pollster::block_on(validation_scope.pop());
        let memory_error = pollster::block_on(memory_scope.pop());
        if let Some(error) = validation_error.or(memory_error) {
            return Err(unavailable(error));
        }
        poll.map_err(unavailable)?;
        receiver
            .try_recv()
            .map_err(unavailable)?
            .map_err(unavailable)?;
        let mapped = slice.get_mapped_range().map_err(unavailable)?;
        let output = bytemuck::try_cast_slice::<u8, f32>(&mapped)
            .map_err(unavailable)?
            .to_vec();
        Ok(Tensor::from_f32(
            vec![matrix[0] as usize, matrix[2] as usize],
            output,
        )?)
    }
}

impl RuntimeProvider for WgpuBackend {
    fn name(&self) -> &str {
        "wgpu"
    }

    fn capabilities(&self) -> &[KernelCapability] {
        &self.capabilities
    }
}

impl KernelBackend for WgpuBackend {
    fn reuse_workspace(&self) -> bool {
        self.cpu.reuse_workspace()
    }

    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        self.multiply(lhs, Weights::F32(rhs), false)
    }

    fn matmul_q8_0(&self, lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor> {
        self.multiply(lhs, Weights::Q8(rhs), false)
    }

    fn matmul_q4_0(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        self.multiply(lhs, Weights::Q4(rhs), false)
    }

    fn matmul_q8_0_output_rows(&self, lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor> {
        self.multiply(lhs, Weights::Q8(rhs), true)
    }

    fn matmul_q4_0_output_rows(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        self.multiply(lhs, Weights::Q4(rhs), true)
    }

    fn matmul_q4_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ4K) -> Result<Tensor> {
        self.cpu.matmul_q4_k(lhs, rhs)
    }

    fn matmul_q5_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ5K) -> Result<Tensor> {
        self.cpu.matmul_q5_k(lhs, rhs)
    }

    fn matmul_q6_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ6K) -> Result<Tensor> {
        self.cpu.matmul_q6_k(lhs, rhs)
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
        self.cpu.rms_norm(input, weight, epsilon)
    }

    fn rms_norm_heads(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
        self.cpu.rms_norm_heads(input, weight, epsilon)
    }

    fn softmax(&self, input: &Tensor) -> Result<Tensor> {
        self.cpu.softmax(input)
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        self.cpu.silu_mul(gate, up)
    }

    fn apply_rope(&self, input: &Tensor, cos: &[f32], sin: &[f32]) -> Result<Tensor> {
        self.cpu.apply_rope(input, cos, sin)
    }

    fn embedding(&self, table: &Tensor, token_ids: &[usize]) -> Result<Tensor> {
        self.cpu.embedding(table, token_ids)
    }
}

enum Weights<'a> {
    F32(&'a Tensor),
    Q4(&'a QuantizedTensor4),
    Q8(&'a QuantizedTensor),
}

impl<'a> Weights<'a> {
    fn shape(&self) -> &Shape {
        match self {
            Self::F32(tensor) => tensor.shape(),
            Self::Q4(tensor) => tensor.shape(),
            Self::Q8(tensor) => tensor.shape(),
        }
    }

    fn encoding(&self) -> u32 {
        match self {
            Self::F32(_) => 0,
            Self::Q4(_) => 1,
            Self::Q8(_) => 2,
        }
    }

    fn byte_len(&self) -> Result<u64> {
        let (count, bytes) = match self {
            Self::F32(tensor) => (tensor.as_f32()?.len(), 4),
            Self::Q4(tensor) => (tensor.blocks().len(), 20),
            Self::Q8(tensor) => (tensor.blocks().len(), 36),
        };
        Ok((count as u64) * bytes)
    }

    fn bytes(self) -> Result<Cow<'a, [u8]>> {
        match self {
            Self::F32(tensor) => Ok(Cow::Borrowed(bytemuck::cast_slice(tensor.as_f32()?))),
            Self::Q4(tensor) => {
                let mut bytes = Vec::with_capacity(tensor.blocks().len() * 20);
                for block in tensor.blocks() {
                    if block.values.iter().any(|&code| code > 15) {
                        return Err(invalid("Q4_0 codes must fit in four bits"));
                    }
                    bytes.extend_from_slice(&block.scale.to_le_bytes());
                    bytes.extend(
                        block
                            .values
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|[low, high]| low | (high << 4)),
                    );
                }
                Ok(Cow::Owned(bytes))
            }
            Self::Q8(tensor) => {
                let mut bytes = Vec::with_capacity(tensor.blocks().len() * 36);
                for block in tensor.blocks() {
                    bytes.extend_from_slice(&block.scale.to_le_bytes());
                    bytes.extend(block.values.map(|code| code as u8));
                }
                Ok(Cow::Owned(bytes))
            }
        }
    }
}

fn capabilities() -> Vec<KernelCapability> {
    let capability = |dtype, quantization, layout: &str, operation, alignment| KernelCapability {
        device: Device::Gpu,
        dtype,
        quantization,
        tensor_layout: layout.to_owned(),
        supported_operations: BTreeSet::from([operation]),
        alignment,
        head_dimension: None,
        maximum_sequence_length: None,
    };
    let mut capabilities = vec![capability(
        DType::F32,
        QuantizationScheme::None,
        "row_major",
        Operation::Matmul,
        1,
    )];
    for (dtype, scheme) in [
        (DType::Q4_0, QuantizationScheme::Q4_0),
        (DType::Q8_0, QuantizationScheme::Q8_0),
    ] {
        for layout in ["row_major", GGML_OUTPUT_ROWS_LAYOUT] {
            capabilities.push(capability(
                dtype,
                scheme,
                layout,
                Operation::QuantizedMatmul,
                32,
            ));
        }
    }
    capabilities
}

fn shader_size(dimension: usize) -> Result<u32> {
    u32::try_from(dimension).map_err(|_| unavailable("matrix dimension exceeds shader indexing"))
}

fn shader_elements(rows: u32, columns: u32) -> Result<u32> {
    rows.checked_mul(columns)
        .ok_or_else(|| unavailable("matrix element count exceeds shader indexing"))
}

fn invalid(message: &str) -> KernelError {
    KernelError::InvalidInput(format!("WGPU {message}"))
}

fn unavailable(error: impl std::fmt::Display) -> KernelError {
    KernelError::Unsupported(format!("WGPU {error}"))
}
