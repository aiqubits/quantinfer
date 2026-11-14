//! GGUF writer and streaming spool.

use std::collections::BTreeMap;
use std::io::{self, Read, Seek, Write};

use half::{bf16, f16};
use quantinfer_core::{
    QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K,
    Q4_0_BLOCK_SIZE,
};

use super::encoding::{align_up, encode_metadata, encode_tensor_info};
use super::types::{FormatError, GgmlType, GgufValue, GgufWriter, PendingTensor};
use super::{Result, DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, QUANT_BLOCK_LAYOUT};

impl GgufWriter {
    pub fn new() -> Self {
        Self {
            alignment: DEFAULT_ALIGNMENT,
            ..Self::default()
        }
    }

    pub fn insert_metadata(&mut self, key: impl Into<String>, value: GgufValue) {
        self.metadata.insert(key.into(), value);
    }

    pub fn add_f32_tensor(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(FormatError::Invalid(
                "GGUF tensor names must not be empty".to_owned(),
            ));
        }
        let shape = shape.into();
        if shape.contains(&0) {
            return Err(FormatError::Invalid(
                "F32 tensor dimensions must be greater than zero".to_owned(),
            ));
        }
        let elements = shape.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| FormatError::Invalid("tensor shape overflows usize".to_owned()))
        })?;
        if elements != values.len() {
            return Err(FormatError::Invalid(
                "F32 tensor shape does not match data length".to_owned(),
            ));
        }
        let data_size = values
            .len()
            .checked_mul(4)
            .ok_or_else(|| FormatError::Invalid("F32 tensor size overflows usize".to_owned()))?;
        let mut data = Vec::with_capacity(data_size);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        self.tensors.push(PendingTensor {
            name,
            shape,
            dtype: GgmlType::F32,
            data,
        });
        Ok(())
    }

    pub fn add_f16_tensor(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(FormatError::Invalid(
                "GGUF tensor names must not be empty".to_owned(),
            ));
        }
        let shape = shape.into();
        if shape.contains(&0) {
            return Err(FormatError::Invalid(
                "F16 tensor dimensions must be greater than zero".to_owned(),
            ));
        }
        let elements = shape.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| FormatError::Invalid("tensor shape overflows usize".to_owned()))
        })?;
        if elements != values.len() {
            return Err(FormatError::Invalid(
                "F16 tensor shape does not match data length".to_owned(),
            ));
        }
        let data_size = values
            .len()
            .checked_mul(std::mem::size_of::<f16>())
            .ok_or_else(|| FormatError::Invalid("F16 tensor size overflows usize".to_owned()))?;
        let mut data = Vec::with_capacity(data_size);
        for value in values {
            data.extend_from_slice(&f16::from_f32(*value).to_le_bytes());
        }
        self.tensors.push(PendingTensor {
            name,
            shape,
            dtype: GgmlType::F16,
            data,
        });
        Ok(())
    }

    pub fn add_bf16_tensor(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<()> {
        let name = name.into();
        if name.is_empty() {
            return Err(FormatError::Invalid(
                "GGUF tensor names must not be empty".to_owned(),
            ));
        }
        let shape = shape.into();
        if shape.contains(&0) {
            return Err(FormatError::Invalid(
                "BF16 tensor dimensions must be greater than zero".to_owned(),
            ));
        }
        let elements = shape.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| FormatError::Invalid("tensor shape overflows usize".to_owned()))
        })?;
        if elements != values.len() {
            return Err(FormatError::Invalid(
                "BF16 tensor shape does not match data length".to_owned(),
            ));
        }
        let data_size = values
            .len()
            .checked_mul(std::mem::size_of::<bf16>())
            .ok_or_else(|| FormatError::Invalid("BF16 tensor size overflows usize".to_owned()))?;
        let mut data = Vec::with_capacity(data_size);
        for value in values {
            data.extend_from_slice(&bf16::from_f32(*value).to_le_bytes());
        }
        self.tensors.push(PendingTensor {
            name,
            shape,
            dtype: GgmlType::BF16,
            data,
        });
        Ok(())
    }

    pub fn add_q8_0_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensor) {
        let mut data = Vec::with_capacity(tensor.blocks().len() * 34);
        for block in tensor.blocks() {
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
            data.extend(block.values.iter().map(|value| *value as u8));
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q8_0,
            data,
        });
    }

    pub fn add_q4_0_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensor4) {
        self.insert_metadata(QUANT_BLOCK_LAYOUT, GgufValue::String("ggml".to_owned()));
        let mut data = Vec::with_capacity(tensor.blocks().len() * 18);
        for block in tensor.blocks() {
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
            let (lower, upper) = block.values.split_at(Q4_0_BLOCK_SIZE / 2);
            data.extend(lower.iter().zip(upper).map(|(low, high)| low | (high << 4)));
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q4_0,
            data,
        });
    }

    pub fn add_q4_k_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensorQ4K) {
        let mut data = Vec::with_capacity(tensor.blocks().len() * 144);
        for block in tensor.blocks() {
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
            data.extend_from_slice(&f16::from_f32(block.min_scale).to_le_bytes());
            data.extend_from_slice(&block.scales);
            data.extend_from_slice(&block.values);
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q4_K,
            data,
        });
    }

    pub fn add_q5_k_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensorQ5K) {
        let mut data = Vec::with_capacity(tensor.blocks().len() * 176);
        for block in tensor.blocks() {
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
            data.extend_from_slice(&f16::from_f32(block.min_scale).to_le_bytes());
            data.extend_from_slice(&block.scales);
            data.extend_from_slice(&block.high_bits);
            data.extend_from_slice(&block.values);
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q5_K,
            data,
        });
    }

    pub fn add_q6_k_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensorQ6K) {
        self.insert_metadata(QUANT_BLOCK_LAYOUT, GgufValue::String("ggml".to_owned()));
        let mut data = Vec::with_capacity(tensor.blocks().len() * 210);
        for block in tensor.blocks() {
            data.extend_from_slice(&block.low_bits);
            data.extend_from_slice(&block.high_bits);
            data.extend(block.scales.iter().map(|value| *value as u8));
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q6_K,
            data,
        });
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        let mut header = GgufHeader::new(self.alignment);
        let padding = self
            .tensors
            .iter()
            .map(|tensor| header.append(tensor))
            .collect::<Result<Vec<_>>>()?;
        let data_size = usize::try_from(header.data_size)
            .map_err(|_| FormatError::Invalid("GGUF data size overflows usize".to_owned()))?;
        let mut output = header.encode(self.metadata)?;
        output.try_reserve_exact(data_size).map_err(|error| {
            FormatError::Invalid(format!("cannot allocate GGUF output: {error}"))
        })?;
        for (tensor, padding) in self.tensors.into_iter().zip(padding) {
            output.resize(output.len() + padding, 0);
            output.extend_from_slice(&tensor.data);
        }
        Ok(output)
    }
}

struct GgufHeader {
    names: std::collections::BTreeSet<String>,
    infos: Vec<u8>,
    alignment: u64,
    data_size: u64,
}

impl GgufHeader {
    fn new(alignment: u64) -> Self {
        Self {
            names: Default::default(),
            infos: Vec::new(),
            alignment,
            data_size: 0,
        }
    }

    fn append(&mut self, tensor: &PendingTensor) -> Result<usize> {
        if !self.names.insert(tensor.name.clone()) {
            return Err(FormatError::Invalid(format!(
                "duplicate GGUF tensor name {}",
                tensor.name
            )));
        }
        let offset = align_up(self.data_size, self.alignment)?;
        let padding = (offset - self.data_size) as usize;
        encode_tensor_info(&mut self.infos, tensor, offset)?;
        self.data_size = offset
            .checked_add(tensor.data.len() as u64)
            .ok_or_else(|| FormatError::Invalid("GGUF data size overflows u64".to_owned()))?;
        Ok(padding)
    }

    fn encode(self, mut metadata: BTreeMap<String, GgufValue>) -> Result<Vec<u8>> {
        metadata
            .entry("general.alignment".to_owned())
            .or_insert(GgufValue::U32(self.alignment as u32));
        let mut output = Vec::new();
        output.extend_from_slice(GGUF_MAGIC);
        output.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        output.extend_from_slice(&(self.names.len() as u64).to_le_bytes());
        output.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        output.extend_from_slice(&encode_metadata(&metadata)?);
        output.extend_from_slice(&self.infos);
        let data_start = usize::try_from(align_up(output.len() as u64, self.alignment)?)
            .map_err(|_| FormatError::Invalid("GGUF data offset overflows usize".to_owned()))?;
        output.resize(data_start, 0);
        Ok(output)
    }
}

/// Writes tensor payloads to a scratch stream while retaining only the GGUF header.
pub struct GgufSpool<F> {
    scratch: F,
    header: Option<GgufHeader>,
    writer: GgufWriter,
}

impl<F: Read + Seek + Write> GgufSpool<F> {
    /// The scratch stream must be empty and positioned at its beginning.
    pub fn new(scratch: F, writer: GgufWriter) -> Self {
        Self {
            scratch,
            header: Some(GgufHeader::new(writer.alignment)),
            writer,
        }
    }

    /// Encode a tensor with the usual writer and immediately spool its payload.
    pub fn append(&mut self, encode: impl FnOnce(&mut GgufWriter) -> Result<()>) -> Result<()> {
        let mut header = self.header.take().ok_or_else(|| {
            FormatError::Invalid("cannot continue a failed GGUF stream".to_owned())
        })?;
        encode(&mut self.writer)?;
        for tensor in self.writer.tensors.drain(..) {
            let padding = header.append(&tensor)?;
            self.scratch
                .write_all(&[0; DEFAULT_ALIGNMENT as usize][..padding])
                .and_then(|()| self.scratch.write_all(&tensor.data))
                .map_err(|error| FormatError::Io(format!("cannot spool GGUF tensor: {error}")))?;
        }
        self.header = Some(header);
        Ok(())
    }

    pub fn finish(mut self, output: &mut impl Write) -> Result<()> {
        self.append(|_| Ok(()))?;
        let header = self.header.take().expect("append validated the header");
        let data_size = header.data_size;
        let bytes = header.encode(self.writer.metadata)?;
        let copied = self
            .scratch
            .rewind()
            .and_then(|()| output.write_all(&bytes))
            .and_then(|()| io::copy(&mut self.scratch.take(data_size), output))
            .map_err(|error| FormatError::Io(format!("cannot write GGUF: {error}")))?;
        if copied != data_size {
            return Err(FormatError::Io(
                "GGUF scratch stream was truncated".to_owned(),
            ));
        }
        Ok(())
    }
}
