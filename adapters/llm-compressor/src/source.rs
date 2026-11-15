//! SafeTensors sources for compressed-tensors weights.

use quantinfer_format::{SafeTensorFile, SafeTensorInfo, SafeTensorShards};

pub trait LlmCompressorTensorSource {
    fn tensor_info(&self, name: &str) -> Option<SafeTensorInfo>;
    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String>;
    fn tensor_u32(&self, name: &str) -> Result<Vec<u32>, String>;
    fn tensor_u64(&self, name: &str) -> Result<Vec<u64>, String>;
}

impl LlmCompressorTensorSource for SafeTensorFile {
    fn tensor_info(&self, name: &str) -> Option<SafeTensorInfo> {
        self.tensors().get(name).cloned()
    }

    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        SafeTensorFile::tensor_f32(self, name).map_err(|error| error.to_string())
    }

    fn tensor_u32(&self, name: &str) -> Result<Vec<u32>, String> {
        SafeTensorFile::tensor_u32(self, name).map_err(|error| error.to_string())
    }

    fn tensor_u64(&self, name: &str) -> Result<Vec<u64>, String> {
        SafeTensorFile::tensor_u64(self, name).map_err(|error| error.to_string())
    }
}

impl LlmCompressorTensorSource for SafeTensorShards {
    fn tensor_info(&self, name: &str) -> Option<SafeTensorInfo> {
        self.tensors()
            .find_map(|(tensor_name, info)| (tensor_name == name).then(|| info.clone()))
    }

    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        SafeTensorShards::tensor_f32(self, name).map_err(|error| error.to_string())
    }

    fn tensor_u32(&self, name: &str) -> Result<Vec<u32>, String> {
        SafeTensorShards::tensor_u32(self, name).map_err(|error| error.to_string())
    }

    fn tensor_u64(&self, name: &str) -> Result<Vec<u64>, String> {
        SafeTensorShards::tensor_u64(self, name).map_err(|error| error.to_string())
    }
}
