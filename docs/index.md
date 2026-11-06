# Documentation

## How to use the software

Installation and and get started are described in the [Readme](../Readme.md).

Each reconstruction follows these steps:

1. Configuration. This step loads the config files and selects the input data type. The corresponding command is [`load_config(...)`](usage/configuration/configuration.md).
2. Runime setup. This step  is needed to select compute devices and create the run state. Call `runtime = initialize_runtime(params)`.
3. Data preparation. On this step, managed with [DataPreparer class](usage/data_preparer.md), the data is created or loaded from a specified file, and processed in order to extract all necessary variables needed for the joint reconstruction. Those include the binned k-space and motion data, coil sensitivity maps. If the synthetic input data was selected, DataPreparer also handles the sampling and motion simulation.
5. Joint reconstruction. [JointReconstructor](usage/joint_reconstructor.md) is the class running the GRICS reconstruction itself. It returns motion-corrected image or volume and estimated motion model. It automatically generates some outputs. See [outputs](usage/outputs.md) for the run layout and saved files.
6. Optional post-processing or dicomization (see [post-processing](usage/postprocessing.md)).

## Usage

- [Configuration](usage/configuration/configuration.md)
    - [Input formats](usage/configuration/input_formats.md)
    - [Sampling](usage/configuration/sampling.md)
    - [Motion](usage/configuration/motion.md)
- [Using DataPreparer](usage/data_preparer.md)
- [Using JointReconstructor](usage/joint_reconstructor.md)
- [Postprocessing and export](usage/postprocessing.md)

- [Outputs](usage/outputs.md)
- [Tests and simulation validation](usage/tests.md)
- [Pipeline architecture](pipelines.md) for the reconstruction data flow and the available preimplemented pipelines.

## Shared conventions

- [Data and coordinate conventions](conventions.md) defines tensor layouts, sampling-index order, motion components, and units.

## Architecture

- [Configuration](architecture/configuration.md)
- [Input data loading](architecture/input_data_loading.md)
- [Data preparation](architecture/data_preparation.md)
- [Reconstruction](architecture/reconstruction.md)

## Implementation details

- [JointReconstructor: updates and regularization normalization](implementation/joint_reconstructor.md)
- [ConjugateGradientSolver: regularization, convergence, and stagnation](implementation/conjugate_gradient_solver.md)
- [MotionSimulator: synthetic acquisition and temporal models](implementation/motion_simulator.md)
- [MotionBinner: state selection and representative signals](implementation/motion_binner.md)
- [CoilSensitivityCalculator: calibration, smoothing, and normalization](implementation/coil_sensitivity_calculator.md)
- [MotionOperator: interpolation, boundaries, and adjoint](implementation/motion_operator.md)
- [EncodingOperator: shared-image encoding and faster normal operations](implementation/encoding_operator.md)
- [MotionPerturbationSimulator: Jacobian approximation and boundary gradients](implementation/motion_perturbation_simulator.md)
- [PseudoGaussSeidelMotionPreconditioner: local factors and CPU execution](implementation/motion_preconditioner.md)


