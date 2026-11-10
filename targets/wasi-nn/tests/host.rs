#![cfg(feature = "conformance")]

use quantinfer_target_wasi_nn::{QuantInferBackend, WasiNnLimits};
use wasmtime::{Engine, Linker, Module, Store};
use wasmtime_wasi_nn::backend::{BackendInner, Id, NamedTensor};
use wasmtime_wasi_nn::wit::{ExecutionTarget, Tensor, TensorType};
use wasmtime_wasi_nn::{Backend, Graph, GraphRegistry, Registry};

struct SmolRegistry(Graph);

impl GraphRegistry for SmolRegistry {
    fn get(&self, name: &str) -> Option<&Graph> {
        (name == "smol").then_some(&self.0)
    }

    fn get_mut(&mut self, name: &str) -> Option<&mut Graph> {
        (name == "smol").then_some(&mut self.0)
    }
}

#[test]
fn validates_builders_limits_and_execution_targets() {
    assert!(QuantInferBackend::new(WasiNnLimits {
        generation_tokens: 0,
        ..WasiNnLimits::default()
    })
    .is_err());
    let mut backend = QuantInferBackend::new(WasiNnLimits {
        model_bytes: 1,
        ..WasiNnLimits::default()
    })
    .unwrap();
    assert!(backend.load(&[], ExecutionTarget::Cpu).is_err());
    assert!(backend.load(&[b"GGUF"], ExecutionTarget::Cpu).is_err());
    assert!(backend.load(&[b"GGUF"], ExecutionTarget::Gpu).is_err());
}

#[test]
#[ignore = "requires a pinned SmolLM2 artifact in QUANTINFER_WASI_NN_GGUF"]
fn generates_smollm2_tokens_from_a_wasm_guest() {
    let path = std::env::var_os("QUANTINFER_WASI_NN_GGUF").expect("pinned SmolLM2 GGUF path");
    let backend = QuantInferBackend::default();
    let graph = backend.load_path(path).unwrap();
    let registry = Registry::from(SmolRegistry(graph.clone()));
    let context = wasmtime_wasi_nn::witx::WasiNnCtx::new([Backend::from(backend)], registry);
    let engine = Engine::default();
    let module = Module::new(&engine, include_str!("generate.wat")).unwrap();
    let mut linker = Linker::new(&engine);
    wasmtime_wasi_nn::witx::add_to_linker(&mut linker, |state| state).unwrap();
    let mut store = Store::new(&engine, context);
    let instance = linker.instantiate(&mut store, &module).unwrap();
    let generate = instance
        .get_typed_func::<(), u32>(&mut store, "generate")
        .unwrap();
    assert_eq!(generate.call(&mut store, ()).unwrap(), 32);
    let mut bytes = [0; 32];
    instance
        .get_memory(&mut store, "memory")
        .unwrap()
        .read(&store, 128, &mut bytes)
        .unwrap();
    let expected = [260, 3575, 282, 260, 1798, 30, 198, 198];
    let tokens = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(i32::from_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(tokens, expected);

    let mut named = graph.init_execution_context().unwrap();
    let prompt = b"The capital of France is";
    let inputs = vec![
        NamedTensor {
            name: "prompt".to_owned(),
            tensor: Tensor::new(vec![prompt.len() as u32], TensorType::U8, prompt.to_vec()),
        },
        NamedTensor {
            name: "max_new_tokens".to_owned(),
            tensor: Tensor::new(vec![1], TensorType::I32, 8_i32.to_le_bytes().to_vec()),
        },
    ];
    let outputs = named.compute_with_io(inputs).unwrap();
    assert_eq!(outputs[0].name, "tokens");
    assert_eq!(outputs[0].tensor.data, bytes);
    assert_eq!(outputs[1].name, "text");
    assert_eq!(outputs[1].tensor.data, b" the capital of the country.\n\n");
    assert!(named.compute_with_io(Vec::new()).is_err());
    assert!(named.get_output(Id::Index(0)).is_err());
}
