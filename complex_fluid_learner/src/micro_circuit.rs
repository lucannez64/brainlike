use anyhow::Result;
// No longer need `wgpu` or `bytemuck` imports here
use ndarray::Array1;
use rand::prelude::*;
// No longer need `std::sync::Arc`

// This struct is still needed to provide initial random weights/biases
// for ALL circuits, which are then copied to the global GPU buffers.
// The `CircuitParams` struct is moved to neural_network.rs or directly into batch kernels.
pub struct MicroCircuitInitializer {
    pub n_internal_units: usize,
    pub internal_weights: Array1<f32>, // Store CPU copies for initial data
    pub internal_biases: Array1<f32>,  // Store CPU copies for initial data
}

impl MicroCircuitInitializer {
    pub fn new(
        n_internal_units: usize,
        input_scale: f32,
        _circuit_offset: usize, // No longer directly used here
                                // gpu_context: Arc<GpuContext>, // No longer passed here
    ) -> Result<Self> {
        let mut rng = thread_rng();
        let internal_weights: Array1<f32> =
            Array1::from_shape_fn(n_internal_units, |_| rng.gen_range(-1.0..1.0)) * input_scale;

        let internal_biases: Array1<f32> =
            Array1::from_shape_fn(n_internal_units, |_| rng.gen_range(-1.0..1.0) * 0.5);

        Ok(Self {
            n_internal_units,
            internal_weights,
            internal_biases,
        })
    }
    // No more `activate` or `compute_derivative` methods here.
    // Their logic will be handled by batch kernels in `ComplexFluidLearner`.
}
