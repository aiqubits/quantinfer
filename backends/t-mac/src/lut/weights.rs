use quantinfer_core::QuantizedTensor4;
use quantinfer_kernel::{KernelError, Result};

/// Immutable packed snapshot. Mutating the source Core tensor later does not
/// change this prepared object; applications may share it across requests.
#[derive(Debug, Clone)]
pub struct Lut4Weights {
    pub(super) k: usize,
    pub(super) n: usize,
    pub(super) planes: Vec<u16>,
    pub(super) scales: Vec<f32>,
}

impl Lut4Weights {
    pub fn from_output_rows(rhs: &QuantizedTensor4) -> Result<Self> {
        let [k, n] = rhs.shape().dims() else {
            return Err(KernelError::InvalidInput(
                "LUT4 RHS must be a matrix".into(),
            ));
        };
        if *k == 0 || *n == 0 || !k.is_multiple_of(32) {
            return Err(KernelError::InvalidInput(
                "LUT4 requires nonempty output rows and K divisible by 32".into(),
            ));
        }
        let count = rhs
            .blocks()
            .len()
            .checked_mul(8)
            .ok_or_else(|| KernelError::InvalidInput("LUT4 packed length overflows".into()))?;
        let mut planes = Vec::new();
        let mut scales = Vec::new();
        planes
            .try_reserve_exact(count)
            .map_err(|_| KernelError::InvalidInput("cannot allocate LUT4 bit planes".into()))?;
        scales
            .try_reserve_exact(rhs.blocks().len())
            .map_err(|_| KernelError::InvalidInput("cannot allocate LUT4 scales".into()))?;
        for block in rhs.blocks() {
            if !block.scale.is_finite() || block.values.iter().any(|&code| code > 15) {
                return Err(KernelError::InvalidInput(
                    "LUT4 requires finite scales and valid four-bit codes".into(),
                ));
            }
            scales.push(block.scale);
            for group in block.values.as_chunks::<4>().0 {
                let mut word = 0u16;
                for bit in 0..4 {
                    for (lane, &code) in group.iter().enumerate() {
                        // Offset-binary (code-8) to signed two's-complement planes.
                        word |= u16::from(((code ^ 8) >> bit) & 1) << (bit * 4 + lane);
                    }
                }
                planes.push(word);
            }
        }
        Ok(Self {
            k: *k,
            n: *n,
            planes,
            scales,
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.k, self.n]
    }
    /// Bit-plane/scale payload only; excludes allocator and struct overhead.
    pub fn packed_bytes(&self) -> usize {
        self.planes.len() * size_of::<u16>() + self.scales.len() * size_of::<f32>()
    }
}
