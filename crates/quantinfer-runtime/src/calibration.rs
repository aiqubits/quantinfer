use quantinfer_core::Tensor;
use quantinfer_model::TensorRole;

use crate::Result;

/// Receives a [tokens, channels] activation matrix before a linear projection.
/// Roles that consume the same activations are reported together.
pub trait ProjectionObserver {
    fn observe(&mut self, layer: usize, roles: &[TensorRole], activations: &Tensor) -> Result<()>;
}

impl<F> ProjectionObserver for F
where
    F: FnMut(usize, &[TensorRole], &Tensor) -> Result<()>,
{
    fn observe(&mut self, layer: usize, roles: &[TensorRole], activations: &Tensor) -> Result<()> {
        self(layer, roles, activations)
    }
}

pub(crate) struct ProjectionTrace<'a, 'observer> {
    pub layer: usize,
    pub observer: &'a mut Option<&'observer mut dyn ProjectionObserver>,
}

impl ProjectionTrace<'_, '_> {
    pub fn record(&mut self, roles: &[TensorRole], activations: &Tensor) -> Result<()> {
        if let Some(observer) = self.observer {
            observer.observe(self.layer, roles, activations)?;
        }
        Ok(())
    }
}
