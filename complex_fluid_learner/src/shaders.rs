// In src/shaders.rs

// Existing MICRO_CIRCUIT_FORWARD_SHADER and MICRO_CIRCUIT_BACKWARD_SHADER are no longer used by the batching logic.
// You can remove them if you don't plan to use single-sample processing anymore,
// but I'll keep them commented for now if you want to revert.

// pub const MICRO_CIRCUIT_FORWARD_SHADER: &str = r#" ... "#; // No longer used directly
// pub const MICRO_CIRCUIT_BACKWARD_SHADER: &str = r#" ... "#; // No longer used directly

pub const MATRIX_MULTIPLY_SHADER: &str = r#"
struct MatrixDims {
    m: u32,
    n: u32,
    k: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> dims: MatrixDims;
@group(0) @binding(1) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(2) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(16, 16) // Optimized for common GPU architectures
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    
    var sum = 0.0;
    for (var i = 0u; i < dims.k; i = i + 1u) {
        sum = sum + matrix_a[row * dims.k + i] * matrix_b[i * dims.n + col];
    }
    
    result[row * dims.n + col] = sum;
}
"#;

// NEW BATCHED MICRO-CIRCUIT FORWARD PASS SHADER
pub const BATCH_MICRO_CIRCUIT_FORWARD_SHADER: &str = r#"
struct BatchCircuitParams {
    batch_size: u32,
    n_hidden_circuits: u32,
    n_internal_units_per_circuit: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: BatchCircuitParams;
@group(0) @binding(1) var<storage, read> hidden_circuit_inputs_linear: array<f32>; // [batch_size * n_hidden_circuits]
@group(0) @binding(2) var<storage, read> all_circuits_internal_weights: array<f32>; // [n_hidden_circuits * n_internal_units_per_circuit]
@group(0) @binding(3) var<storage, read> all_circuits_internal_biases: array<f32>; // [n_hidden_circuits * n_internal_units_per_circuit]
@group(0) @binding(4) var<storage, read_write> all_internal_activations: array<f32>; // [batch_size * n_hidden_circuits * n_internal_units_per_circuit]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x; // Global index maps to a specific internal unit within a specific circuit and sample
    let total_elements = params.batch_size * params.n_hidden_circuits * params.n_internal_units_per_circuit;
    if (index >= total_elements) {
        return;
    }
    
    // Deconstruct index: index = sample_idx * H * U + circuit_idx * U + unit_idx
    let unit_idx = index % params.n_internal_units_per_circuit;
    let temp = index / params.n_internal_units_per_circuit;
    let circuit_idx = temp % params.n_hidden_circuits;
    let sample_idx = temp / params.n_hidden_circuits;

    // Get input for this specific sample and circuit
    let circuit_input_scalar = hidden_circuit_inputs_linear[sample_idx * params.n_hidden_circuits + circuit_idx];
    
    // Get internal weights and biases for this specific circuit (same for all samples for this circuit)
    let internal_weights_offset = circuit_idx * params.n_internal_units_per_circuit;
    let internal_biases_offset = circuit_idx * params.n_internal_units_per_circuit;

    let pre_activation = all_circuits_internal_weights[internal_weights_offset + unit_idx] * circuit_input_scalar +
                         all_circuits_internal_biases[internal_biases_offset + unit_idx];
    
    // Compute sigmoid activation
    all_internal_activations[index] = 1.0 / (1.0 + exp(-pre_activation));

    // Note: The mean calculation (to get hidden_circuit_outputs) for each circuit 
    // is a reduction operation. Doing it in this same kernel is complex (requires local memory
    // and multiple passes). For simplicity for now, the mean will be calculated on the CPU
    // after reading back `all_internal_activations`, but a dedicated GPU reduction kernel
    // would be the next step for further optimization.
}
"#;
