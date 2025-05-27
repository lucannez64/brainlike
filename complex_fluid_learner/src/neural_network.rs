use crate::gpu_context::{GpuContext, MatrixDims};
use crate::micro_circuit::MicroCircuitInitializer;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array2, Axis, s};
use rand::prelude::*;
use rand::rngs::StdRng; // Add StdRng for seeded generation
use std::sync::Arc;
use std::time::Instant;

// --- NEW KERNEL PARAM STRUCTS ---
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BatchCircuitParams {
    pub batch_size: u32,
    pub n_hidden_circuits: u32,
    pub n_internal_units_per_circuit: u32,
    pub _padding: u32,
}

// --- COMPLEX FLUID LEARNER ---
pub struct ComplexFluidLearner {
    _n_inputs: usize,
    pub n_outputs: usize,
    pub n_hidden_circuits: usize,
    pub n_internal_units_per_circuit: usize,
    pub learning_rate: f32,

    // Network weights - NOW ON GPU (CPU copies for easy updates/reading if needed)
    w1_gpu: wgpu::Buffer,
    b1_gpu: wgpu::Buffer,
    w2_gpu: wgpu::Buffer,
    b2_gpu: wgpu::Buffer,
    pub w1: Array2<f32>, // Keep CPU copies for host-side updates/reads
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,

    // Micro circuits (now consolidated into global buffers on GPU)
    all_circuits_internal_weights_gpu: wgpu::Buffer,
    all_circuits_internal_biases_gpu: wgpu::Buffer,

    n_internal_units_total: usize, // n_hidden_circuits * n_internal_units_per_circuit

    // GPU context
    _gpu_context: Arc<GpuContext>,

    // Pipelines (compiled once)
    batch_forward_microcircuit_pipeline: wgpu::ComputePipeline,
    add_bias_pipeline: wgpu::ComputePipeline, // For W1 + B1 and W2 + B2

    // Training history
    pub training_history: TrainingHistory,
}

#[derive(Default)]
pub struct TrainingHistory {
    pub loss: Vec<f32>,
    pub val_loss: Vec<f32>,
    pub epochs: Vec<usize>,
}

impl ComplexFluidLearner {
    pub async fn new(
        n_inputs: usize,
        n_outputs: usize,
        n_hidden_circuits: usize,
        n_internal_units_per_circuit: usize,
        learning_rate: f32,
        seed: Option<u64>,
    ) -> Result<Self> {
        // FIX: Ensure rng is StdRng for both branches
        let mut rng: StdRng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            // Fallback to a seeded StdRng from a random source
            StdRng::from_rng(thread_rng())?
        };

        let gpu_context: Arc<GpuContext> = Arc::new(GpuContext::new().await?);

        // Initialize network weights (CPU)
        let limit_w1 = (6.0 / (n_inputs + n_hidden_circuits) as f32).sqrt();
        let w1 = Array2::from_shape_fn((n_inputs, n_hidden_circuits), |_| {
            rng.gen_range(-limit_w1..limit_w1)
        });
        let b1 = Array1::zeros(n_hidden_circuits);

        let limit_w2 = (6.0 / (n_hidden_circuits + n_outputs) as f32).sqrt();
        let w2 = Array2::from_shape_fn((n_hidden_circuits, n_outputs), |_| {
            rng.gen_range(-limit_w2..limit_w2)
        });
        let b2 = Array1::zeros(n_outputs);

        // Transfer initial network weights to GPU
        let w1_gpu = gpu_context.create_buffer_with_data(
            w1.as_slice().unwrap(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let b1_gpu = gpu_context.create_buffer_with_data(
            b1.as_slice().unwrap(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let w2_gpu = gpu_context.create_buffer_with_data(
            w2.as_slice().unwrap(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let b2_gpu = gpu_context.create_buffer_with_data(
            b2.as_slice().unwrap(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        // Initialize ALL micro circuit internal weights/biases (CPU)
        let n_internal_units_total = n_hidden_circuits * n_internal_units_per_circuit;
        let mut all_internal_weights_cpu = Vec::with_capacity(n_internal_units_total);
        let mut all_internal_biases_cpu = Vec::with_capacity(n_internal_units_total);

        for i in 0..n_hidden_circuits {
            let initializer = MicroCircuitInitializer::new(
                n_internal_units_per_circuit,
                1.5,
                i, /* No gpu_context */
            )?;
            all_internal_weights_cpu
                .extend_from_slice(initializer.internal_weights.as_slice().unwrap());
            all_internal_biases_cpu
                .extend_from_slice(initializer.internal_biases.as_slice().unwrap());
        }

        // Transfer all micro circuit weights/biases to ONE GPU buffer
        let all_circuits_internal_weights_gpu = gpu_context.create_buffer_with_data(
            &all_internal_weights_cpu,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let all_circuits_internal_biases_gpu = gpu_context.create_buffer_with_data(
            &all_internal_biases_cpu,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        // --- Compile GPU Pipelines (once at initialization) ---
        // 1. Batch Micro Circuit Forward Pipeline
        let batch_forward_microcircuit_shader =
            gpu_context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Batch Micro Circuit Forward Shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        crate::shaders::BATCH_MICRO_CIRCUIT_FORWARD_SHADER.into(),
                    ),
                });

        let batch_forward_microcircuit_bind_group_layout = gpu_context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }, // BatchCircuitParams
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }, // hidden_circuit_inputs_linear
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }, // all_circuits_internal_weights
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }, // all_circuits_internal_biases
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }, // all_internal_activations_gpu (output)
                       // binding 5 for hidden_circuit_outputs (mean) will be part of a separate reduction kernel
                ],
            });

        let batch_forward_microcircuit_pipeline_layout =
            gpu_context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&batch_forward_microcircuit_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let batch_forward_microcircuit_pipeline =
            gpu_context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Batch Micro Circuit Forward Pipeline"),
                    layout: Some(&batch_forward_microcircuit_pipeline_layout),
                    module: &batch_forward_microcircuit_shader,
                    entry_point: "main",
                });

        // 2. Add Bias Pipeline (for A + B or A + scalar/vector)
        let add_bias_shader = gpu_context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Add Bias Shader"),
            source: wgpu::ShaderSource::Wgsl(r#"
                @group(0) @binding(0) var<storage, read_write> input_matrix: array<f32>; // M x N
                @group(0) @binding(1) var<storage, read> bias_vector: array<f32>;       // N (or scalar)
                @group(0) @binding(2) var<uniform> dims: vec2<u32>; // dims.x = N (cols), dims.y = M (rows)

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    let N = dims.x;
                    let total_size = N * dims.y;
                    if (idx >= total_size) { return; }
                    
                    let col_idx = idx % N;
                    input_matrix[idx] = input_matrix[idx] + bias_vector[col_idx];
                }
            "#.into()),
        });

        let add_bias_bind_group_layout =
            gpu_context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let add_bias_pipeline_layout =
            gpu_context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&add_bias_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let add_bias_pipeline =
            gpu_context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Add Bias Pipeline"),
                    layout: Some(&add_bias_pipeline_layout),
                    module: &add_bias_shader,
                    entry_point: "main",
                });

        Ok(Self {
            _n_inputs: n_inputs,
            n_outputs,
            n_hidden_circuits,
            n_internal_units_per_circuit,
            learning_rate,
            w1,
            b1,
            w2,
            b2, // CPU copies
            w1_gpu,
            b1_gpu,
            w2_gpu,
            b2_gpu, // GPU buffers
            all_circuits_internal_weights_gpu,
            all_circuits_internal_biases_gpu,
            n_internal_units_total,
            _gpu_context: gpu_context,
            batch_forward_microcircuit_pipeline,
            add_bias_pipeline,
            training_history: TrainingHistory::default(),
        })
    }

    // --- BATCH FORWARD PASS ---
    // Takes a batch of inputs and computes batch predictions, entirely on GPU
    pub async fn forward_pass_batch(
        &self,
        inputs_batch: &Array2<f32>,
    ) -> Result<ForwardPassCacheBatch> {
        let batch_size = inputs_batch.nrows();
        let current_input_data_gpu = self._gpu_context.create_buffer_with_data(
            inputs_batch.as_slice().unwrap(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        // 1. Compute hidden layer linear activation (X * W1)
        let hidden_linear_pre_bias_gpu = self
            ._gpu_context
            .gemm_gpu(
                &current_input_data_gpu,
                &self.w1_gpu,
                batch_size as u32,
                self.n_hidden_circuits as u32,
                self._n_inputs as u32,
            )
            .await?;

        // 2. Add B1 bias (X * W1 + B1)
        let hidden_circuit_inputs_linear_gpu = self._gpu_context.create_empty_buffer(
            (batch_size * self.n_hidden_circuits * std::mem::size_of::<f32>()) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC, // This buffer will be modified in-place
        );
        self._gpu_context.queue.submit(Some({
            // Copy data to the target buffer first
            let mut encoder = self
                ._gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(
                &hidden_linear_pre_bias_gpu,
                0,
                &hidden_circuit_inputs_linear_gpu,
                0,
                (batch_size * self.n_hidden_circuits * std::mem::size_of::<f32>()) as u64,
            );
            encoder.finish()
        }));

        let add_bias_bind_group_layout = self.add_bias_pipeline.get_bind_group_layout(0);
        let add_bias_bind_group =
            self._gpu_context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Hidden Bias Add Bind Group"),
                    layout: &add_bias_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: hidden_circuit_inputs_linear_gpu.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.b1_gpu.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self
                                ._gpu_context
                                .create_buffer_with_data(
                                    &[self.n_hidden_circuits as u32, batch_size as u32],
                                    wgpu::BufferUsages::UNIFORM,
                                )
                                .as_entire_binding(),
                        },
                    ],
                });

        self._gpu_context.queue.submit(Some({
            let mut encoder = self
                ._gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Add Bias Compute Pass 1"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.add_bias_pipeline);
                compute_pass.set_bind_group(0, &add_bias_bind_group, &[]);
                let workgroup_size_x = ((batch_size * self.n_hidden_circuits) as u32 + 255) / 256;
                compute_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
            }
            encoder.finish()
        }));

        // 3. Process micro-circuits (activations)
        let all_internal_activations_gpu = self._gpu_context.create_empty_buffer(
            (batch_size
                * self.n_hidden_circuits
                * self.n_internal_units_per_circuit
                * std::mem::size_of::<f32>()) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let batch_params = BatchCircuitParams {
            batch_size: batch_size as u32,
            n_hidden_circuits: self.n_hidden_circuits as u32,
            n_internal_units_per_circuit: self.n_internal_units_per_circuit as u32,
            _padding: 0,
        };

        let batch_params_buffer = self._gpu_context.create_buffer_with_data(
            &[batch_params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let batch_forward_microcircuit_bind_group_layout = self
            .batch_forward_microcircuit_pipeline
            .get_bind_group_layout(0);
        let batch_forward_microcircuit_bind_group =
            self._gpu_context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Batch Micro Circuit Forward Bind Group"),
                    layout: &batch_forward_microcircuit_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: batch_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: hidden_circuit_inputs_linear_gpu.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.all_circuits_internal_weights_gpu.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.all_circuits_internal_biases_gpu.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: all_internal_activations_gpu.as_entire_binding(),
                        },
                    ],
                });

        self._gpu_context.queue.submit(Some({
            let mut encoder = self
                ._gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Batch Micro Circuit Forward Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.batch_forward_microcircuit_pipeline);
                compute_pass.set_bind_group(0, &batch_forward_microcircuit_bind_group, &[]);
                let total_internal_units_per_batch =
                    batch_size * self.n_hidden_circuits * self.n_internal_units_per_circuit;
                let workgroup_size_x = (total_internal_units_per_batch as u32 + 255) / 256;
                compute_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
            }
            encoder.finish()
        }));

        // 4. Calculate hidden circuit outputs (mean of internal activations)
        // This is a reduction. For simplicity, we'll read back `all_internal_activations_gpu`
        // and compute the means on the CPU. A proper GPU solution would involve another kernel for reduction.
        let all_internal_activations_cpu_flat = self
            ._gpu_context
            .read_buffer::<f32>(
                &all_internal_activations_gpu,
                (batch_size * self.n_hidden_circuits * self.n_internal_units_per_circuit) as usize,
            )
            .await?;

        let mut hidden_circuit_outputs_cpu = Array2::zeros((batch_size, self.n_hidden_circuits));
        for s_idx in 0..batch_size {
            for h_idx in 0..self.n_hidden_circuits {
                let start =
                    (s_idx * self.n_hidden_circuits + h_idx) * self.n_internal_units_per_circuit;
                let end = start + self.n_internal_units_per_circuit;
                let slice = &all_internal_activations_cpu_flat[start..end];
                let sum: f32 = slice.iter().sum();
                hidden_circuit_outputs_cpu[[s_idx, h_idx]] =
                    sum / (self.n_internal_units_per_circuit as f32);
            }
        }
        let hidden_circuit_outputs_gpu = self._gpu_context.create_buffer_with_data(
            // Transfer back to GPU
            hidden_circuit_outputs_cpu.as_slice().unwrap(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );

        // 5. Compute final output linear activation (Hidden_Outputs * W2)
        let final_linear_pre_bias_gpu = self
            ._gpu_context
            .gemm_gpu(
                &hidden_circuit_outputs_gpu,
                &self.w2_gpu,
                batch_size as u32,
                self.n_outputs as u32,
                self.n_hidden_circuits as u32,
            )
            .await?;

        // 6. Add B2 bias (Hidden_Outputs * W2 + B2)
        let final_prediction_gpu = self._gpu_context.create_empty_buffer(
            (batch_size * self.n_outputs * std::mem::size_of::<f32>()) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        self._gpu_context.queue.submit(Some({
            // Copy data to the target buffer first
            let mut encoder = self
                ._gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(
                &final_linear_pre_bias_gpu,
                0,
                &final_prediction_gpu,
                0,
                (batch_size * self.n_outputs * std::mem::size_of::<f32>()) as u64,
            );
            encoder.finish()
        }));

        let add_bias_bind_group =
            self._gpu_context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Output Bias Add Bind Group"),
                    layout: &add_bias_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: final_prediction_gpu.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.b2_gpu.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self
                                ._gpu_context
                                .create_buffer_with_data(
                                    &[self.n_outputs as u32, batch_size as u32],
                                    wgpu::BufferUsages::UNIFORM,
                                )
                                .as_entire_binding(),
                        },
                    ],
                });

        self._gpu_context.queue.submit(Some({
            let mut encoder = self
                ._gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Add Bias Compute Pass 2"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.add_bias_pipeline);
                compute_pass.set_bind_group(0, &add_bias_bind_group, &[]);
                let workgroup_size_x = ((batch_size * self.n_outputs) as u32 + 255) / 256;
                compute_pass.dispatch_workgroups(workgroup_size_x, 1, 1);
            }
            encoder.finish()
        }));

        // Read final prediction back to CPU for loss calculation (for now)
        let final_prediction_cpu = self
            ._gpu_context
            .read_buffer::<f32>(
                &final_prediction_gpu,
                (batch_size * self.n_outputs) as usize,
            )
            .await?;
        let final_prediction_array =
            Array2::from_shape_vec((batch_size, self.n_outputs), final_prediction_cpu)?;

        Ok(ForwardPassCacheBatch {
            inputs_batch: inputs_batch.clone(),
            hidden_circuit_outputs_gpu,
            all_internal_activations_gpu,
            final_prediction_gpu, // Store GPU buffer for backward pass
            final_prediction_array,
        })
    }

    // --- BATCH BACKWARD PASS (Placeholder for now) ---
    // The backward pass is significantly more complex to fully batch on GPU.
    // For this response, we'll keep the `backward_pass_batch` largely CPU-based,
    // to allow the forward pass improvements to be tested first.
    // A fully GPU-accelerated backward pass would require multiple complex kernels
    // (e.g., matrix transpose, element-wise ops, scatter-gather, and batched reduction).
    pub async fn backward_pass_batch(
        &self,
        cache: &ForwardPassCacheBatch,
        targets_batch: &Array2<f32>,
    ) -> Result<GradientsBatch> {
        let batch_size = targets_batch.nrows();

        let mut dw2_batch_cpu = Array2::zeros((self.n_hidden_circuits, self.n_outputs));
        let mut db2_batch_cpu = Array1::zeros(self.n_outputs);
        let mut dw1_batch_cpu = Array2::zeros((self._n_inputs, self.n_hidden_circuits));
        let mut db1_batch_cpu = Array1::zeros(self.n_hidden_circuits);

        // Read necessary data from GPU to CPU for backward pass (this is a bottleneck)
        let hidden_circuit_outputs_cpu_flat = self
            ._gpu_context
            .read_buffer::<f32>(
                &cache.hidden_circuit_outputs_gpu,
                (batch_size * self.n_hidden_circuits) as usize,
            )
            .await?;
        let hidden_circuit_outputs_cpu = Array2::from_shape_vec(
            (batch_size, self.n_hidden_circuits),
            hidden_circuit_outputs_cpu_flat,
        )?;

        let all_internal_activations_cpu_flat = self
            ._gpu_context
            .read_buffer::<f32>(
                &cache.all_internal_activations_gpu,
                (batch_size * self.n_hidden_circuits * self.n_internal_units_per_circuit) as usize,
            )
            .await?;

        // Simplified backward pass loop on CPU for each sample in the batch
        for i in 0..batch_size {
            let pred_i = cache.final_prediction_array.row(i).to_owned();
            let target_i = targets_batch.row(i).to_owned();
            let hidden_out_i = hidden_circuit_outputs_cpu.row(i).to_owned();
            let input_i = cache.inputs_batch.row(i).to_owned();

            let error_output_layer = &pred_i - &target_i;

            // Gradients for W2 and B2
            dw2_batch_cpu = &dw2_batch_cpu
                + &hidden_out_i
                    .insert_axis(Axis(1))
                    .dot(&error_output_layer.clone().insert_axis(Axis(0))); // FIX: Clone error_output_layer
            db2_batch_cpu = &db2_batch_cpu + &error_output_layer;

            // Backpropagate error to hidden layer
            let hidden_error = error_output_layer.dot(&self.w2.t()); // self.w2 is CPU copy

            // Compute derivatives for each micro circuit (still per-sample, per-circuit on CPU)
            let mut dL_dz_h = Array1::zeros(self.n_hidden_circuits);
            for h_idx in 0..self.n_hidden_circuits {
                let start_internal_activations =
                    (i * self.n_hidden_circuits + h_idx) * self.n_internal_units_per_circuit;
                let end_internal_activations =
                    start_internal_activations + self.n_internal_units_per_circuit;
                let internal_activations_slice = &all_internal_activations_cpu_flat
                    [start_internal_activations..end_internal_activations];
                let internal_weights_start = h_idx * self.n_internal_units_per_circuit;

                let mut sum_weighted_derivatives = 0.0;
                for u_idx in 0..self.n_internal_units_per_circuit {
                    let activated = internal_activations_slice[u_idx];
                    let ds_dz = activated * (1.0 - activated);
                    // Use CPU copy of internal weights
                    let internal_weights_cpu = self
                        ._gpu_context
                        .read_buffer::<f32>(
                            &self.all_circuits_internal_weights_gpu,
                            self.n_internal_units_total,
                        )
                        .await?;
                    sum_weighted_derivatives +=
                        ds_dz * internal_weights_cpu[internal_weights_start + u_idx];
                }
                let circuit_derivative_mean =
                    sum_weighted_derivatives / (self.n_internal_units_per_circuit as f32);
                dL_dz_h[h_idx] = hidden_error[h_idx] * circuit_derivative_mean;
            }

            // Gradients for W1 and B1
            dw1_batch_cpu = &dw1_batch_cpu
                + &input_i
                    .insert_axis(Axis(1))
                    .dot(&dL_dz_h.clone().insert_axis(Axis(0))); // FIX: Clone dL_dz_h
            db1_batch_cpu = &db1_batch_cpu + &dL_dz_h;
        }

        // Average gradients over the batch
        let batch_size_f32 = batch_size as f32;
        Ok(GradientsBatch {
            dw1: dw1_batch_cpu / batch_size_f32,
            db1: db1_batch_cpu / batch_size_f32,
            dw2: dw2_batch_cpu / batch_size_f32,
            db2: db2_batch_cpu / batch_size_f32,
        })
    }

    // --- TRAIN METHOD ---
    pub async fn train(
        &mut self,
        train_x: &Array2<f32>,
        train_y: &Array2<f32>,
        n_epochs: usize,
        batch_size: usize, // Use batch_size now
        validation_split: f32,
        min_epochs_no_improve: usize,
        patience_no_improve: usize,
    ) -> Result<()> {
        let n_samples = train_x.nrows();
        let n_val = (n_samples as f32 * validation_split) as usize;

        let val_x = train_x.slice(s![..n_val, ..]).to_owned();
        let val_y = train_y.slice(s![..n_val, ..]).to_owned();
        let train_x = train_x.slice(s![n_val.., ..]).to_owned();
        let train_y = train_y.slice(s![n_val.., ..]).to_owned();

        let n_train = train_x.nrows();

        let mut best_val_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;
        let mut best_weights: Option<(Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>)> = None;

        println!(
            "Training on {} samples, validating on {} samples",
            n_train, n_val
        );

        let pb = ProgressBar::new(n_epochs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:.cyan/blue}] {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("#>-")
        );

        let training_start_time = Instant::now();

        for epoch in 0..n_epochs {
            let epoch_start_time = Instant::now();
            let mut epoch_loss = 0.0;

            let mut indices: Vec<usize> = (0..n_train).collect();
            indices.shuffle(&mut thread_rng());

            // Process data in batches
            for i in (0..n_train).step_by(batch_size) {
                let current_batch_indices = &indices[i..std::cmp::min(i + batch_size, n_train)];

                let current_batch_x = train_x.select(Axis(0), current_batch_indices);
                let current_batch_y = train_y.select(Axis(0), current_batch_indices);

                // Forward pass for the batch
                let cache = self.forward_pass_batch(&current_batch_x.to_owned()).await?; // .to_owned() to pass ArrayView to Array2

                // Compute loss for the batch
                let batch_loss = (&cache.final_prediction_array - &current_batch_y)
                    .mapv(|x| x * x)
                    .mean()
                    .unwrap();
                epoch_loss += batch_loss * current_batch_x.nrows() as f32; // Accumulate total loss

                // Backward pass for the batch
                let gradients = self
                    .backward_pass_batch(&cache, &current_batch_y.to_owned())
                    .await?;

                // Gradient clipping (done on CPU after batched backward pass)
                let clip_value = 1.0;
                let mut dw1 = gradients.dw1;
                let mut db1 = gradients.db1;
                let mut dw2 = gradients.dw2;
                let mut db2 = gradients.db2;

                dw1.mapv_inplace(|x| x.max(-clip_value).min(clip_value));
                db1.mapv_inplace(|x| x.max(-clip_value).min(clip_value));
                dw2.mapv_inplace(|x| x.max(-clip_value).min(clip_value));
                db2.mapv_inplace(|x| x.max(-clip_value).min(clip_value));

                // Update weights (on CPU, then transfer to GPU)
                self.w1 = &self.w1 - &(dw1 * self.learning_rate);
                self.b1 = &self.b1 - &(db1 * self.learning_rate);
                self.w2 = &self.w2 - &(dw2 * self.learning_rate);
                self.b2 = &self.b2 - &(db2 * self.learning_rate);

                // Update GPU buffers with new CPU weights
                self._gpu_context.queue.write_buffer(
                    &self.w1_gpu,
                    0,
                    bytemuck::cast_slice(self.w1.as_slice().unwrap()),
                );
                self._gpu_context.queue.write_buffer(
                    &self.b1_gpu,
                    0,
                    bytemuck::cast_slice(self.b1.as_slice().unwrap()),
                );
                self._gpu_context.queue.write_buffer(
                    &self.w2_gpu,
                    0,
                    bytemuck::cast_slice(self.w2.as_slice().unwrap()),
                );
                self._gpu_context.queue.write_buffer(
                    &self.b2_gpu,
                    0,
                    bytemuck::cast_slice(self.b2.as_slice().unwrap()),
                );
            }

            // Calculate average epoch loss
            let avg_epoch_loss = epoch_loss / n_train as f32;

            // Validation loss (using batch prediction for validation as well)
            let val_predictions = self.predict_batch(&val_x, batch_size).await?;
            let val_loss = (val_y.clone() - val_predictions)
                .mapv(|x| x * x)
                .mean()
                .unwrap();

            // Update training history
            self.training_history.loss.push(avg_epoch_loss);
            self.training_history.val_loss.push(val_loss);
            self.training_history.epochs.push(epoch);

            // Early stopping logic
            let improvement_marker = if val_loss < best_val_loss {
                best_val_loss = val_loss;
                epochs_without_improvement = 0;
                best_weights = Some((
                    self.w1.clone(),
                    self.b1.clone(),
                    self.w2.clone(),
                    self.b2.clone(),
                ));
                "★"
            } else {
                epochs_without_improvement += 1;
                " "
            };

            // Calculate performance metrics for this epoch
            let epoch_duration = epoch_start_time.elapsed();
            let total_elapsed = training_start_time.elapsed();
            let samples_per_sec = n_train as f32 / epoch_duration.as_secs_f32();

            // Update progress bar message and position
            pb.set_message(format!(
                "Epoch {:4}/{}: Train={:.6}, Val={:.6} {} | {:.1} s/s | {} no-improve",
                epoch + 1,
                n_epochs,
                avg_epoch_loss,
                val_loss,
                improvement_marker,
                samples_per_sec,
                epochs_without_improvement
            ));
            pb.inc(1); // Increment bar position after epoch

            // Early stopping check
            if epoch >= min_epochs_no_improve && epochs_without_improvement >= patience_no_improve {
                pb.finish_with_message(format!(
                    "Early stopping! Best val loss: {:.6} | Total time: {:.1}s",
                    best_val_loss,
                    total_elapsed.as_secs_f32()
                ));

                if let Some((w1, b1, w2, b2)) = best_weights {
                    self.w1 = w1;
                    self.b1 = b1;
                    self.w2 = w2;
                    self.b2 = b2;
                }

                println!(
                    "\n🔄 Restored best weights from epoch {}",
                    epoch + 1 - epochs_without_improvement
                );
                return Ok(());
            }
        }

        let total_time = training_start_time.elapsed();
        pb.finish_with_message(format!(
            "Training complete! Best val loss: {:.6} | Total time: {:.1}s",
            best_val_loss,
            total_time.as_secs_f32()
        ));

        println!("\n✅ Training completed successfully!");
        println!("📊 Final Statistics:");
        println!("   Best validation loss: {:.6}", best_val_loss);
        println!("   Total training time: {:.1}s", total_time.as_secs_f32());
        println!(
            "   Average time per epoch: {:.1}s",
            total_time.as_secs_f32() / n_epochs as f32
        );
        println!("   Total samples processed: {}", n_epochs * n_train);

        Ok(())
    }

    // --- BATCH PREDICT METHOD ---
    // Takes a batch of inputs and returns a batch of predictions
    pub async fn predict_batch(
        &self,
        inputs_batch: &Array2<f32>,
        batch_size: usize,
    ) -> Result<Array2<f32>> {
        let n_samples = inputs_batch.nrows();
        let mut all_predictions = Array2::zeros((n_samples, self.n_outputs));

        for i in (0..n_samples).step_by(batch_size) {
            let current_batch_size = std::cmp::min(batch_size, n_samples - i);
            let current_input_batch = inputs_batch
                .slice(s![i..i + current_batch_size, ..])
                .to_owned();

            let cache = self.forward_pass_batch(&current_input_batch).await?;

            all_predictions
                .slice_mut(s![i..i + current_batch_size, ..])
                .assign(&cache.final_prediction_array);
        }
        Ok(all_predictions)
    }

    // Old single-sample predict, now calls batch predict
    pub async fn predict(&self, inputs: &Array2<f32>, batch_size: usize) -> Result<Array2<f32>> {
        let n_samples = inputs.nrows();
        let pb = ProgressBar::new(n_samples as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:.cyan/blue}] {pos}/{len} ({eta}) | Predicting...")
                .unwrap()
                .progress_chars("#>-")
        );

        let predictions = self.predict_batch(inputs, batch_size).await?;

        pb.set_position(n_samples as u64);
        pb.finish_with_message("Prediction complete!");
        Ok(predictions)
    }
}

// --- BATCH CACHE AND GRADIENTS ---
pub struct ForwardPassCacheBatch {
    pub inputs_batch: Array2<f32>, // Original CPU inputs (needed for dW1 calc)
    pub hidden_circuit_outputs_gpu: wgpu::Buffer, // GPU buffer for hidden layer outputs
    pub all_internal_activations_gpu: wgpu::Buffer, // GPU buffer for all internal activations
    pub final_prediction_gpu: wgpu::Buffer, // GPU buffer for the final prediction
    pub final_prediction_array: Array2<f32>, // Read back to CPU for loss calculation
}

pub struct GradientsBatch {
    pub dw1: Array2<f32>,
    pub db1: Array1<f32>,
    pub dw2: Array2<f32>,
    pub db2: Array1<f32>,
}
