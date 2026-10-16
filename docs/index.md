# Documentation

Start with the [pipeline architecture](pipelines.md) for the reconstruction data flow and the available preimplemented pipelines.

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

## Usage

- [Configuration](usage/configuration.md)
- [Input formats](usage/input_formats.md)
- [Sampling](usage/sampling.md)
- [Motion](usage/motion.md)
- [Outputs](usage/outputs.md)
- [Integration APIs](usage/api.md)

Installation and the shortest path to a first reconstruction remain in the [Readme](../Readme.md).
