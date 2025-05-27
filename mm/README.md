# Multi-Task Complex Learner in Rust

This project implements a multi-task complex learner inspired by a Python implementation. The focus is on leveraging low-level GPU capabilities for efficient computation, excluding the use of the `tch` crate.

## Project Structure

- **Cargo.toml**: The manifest file for Rust's package manager, Cargo. It includes dependencies and project metadata.
- **.gitignore**: Specifies files and directories that should be ignored by Git.
- **src/**: Contains the source code for the project.
  - **main.rs**: The entry point of the application.
  - **ops/**: Contains modules for various operations.
    - **mod.rs**: The module file that declares the submodules.
    - **activation.rs**: Implements activation functions.
    - **loss.rs**: Implements loss functions.
    - **kwta.rs**: Implements k-Winners-Take-All functionality.
  - **gpu/**: Contains GPU-related code.
    - **lib.rs**: The main library file for GPU operations.
    - **context.rs**: Manages GPU context and resources.
    - **kernels.rs**: Contains GPU kernel implementations.

## Features

- **Multi-Task Learning**: Supports multiple regression and classification tasks.
- **GPU Acceleration**: Utilizes low-level GPU programming for performance.
- **Custom Activation Functions**: Implements various activation functions including ReLU and Leaky ReLU.
- **Loss Functions**: Supports multiple loss functions for regression and classification tasks.
- **k-Winners-Take-All (kWTA)**: Implements a mechanism to select the top k activations.

## Getting Started

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mm
   ```

2. Build the project:
   ```
   cargo build
   ```

3. Run the project:
   ```
   cargo run
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.