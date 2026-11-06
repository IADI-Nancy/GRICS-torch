# Documentation

This documentation provides details regarding the core algorithm, implementation details, software usage, conventions and architecture.

## Every reconstruction

Each reconstruction follows these steps:

1. Select the input `data_type`, reconstruction TOML, and any required sampling or motion TOMLs with [`load_config(...)`](usage/configuration.md). The [input data types](usage/input_formats.md), [sampling](usage/sampling.md), and [motion](usage/motion.md) guides define the valid combinations.
2. Call `runtime = initialize_runtime(params)` to select compute devices and create the run state. The [pipeline workflow](pipelines.md#basic-pipeline) shows where runtime initialization belongs.
3. Create `DataPreparer(params=params, runtime=runtime, filename=...)` with the validated parameters, initialized runtime, and input filename (omit the filename for Shepp–Logan data). It converts the selected source into the acquisition interface described by [input data loading](architecture/input_data_loading.md).
4. Call `data.load_input_data()` once, then `data.prepare_reconstructor_inputs(slice_idx=...)` to select and prepare a 2D real-data slice, or omit `slice_idx` for a full 3D volume or synthetic data. Preparation estimates coil maps, applies or imports motion, and builds the sampling layout; see [data preparation](architecture/data_preparation.md).
5. Create `JointReconstructor.from_prepared_data(data)` and call `run()` to perform the configured image and motion reconstruction. The [standard reconstruction workflow](pipelines.md#standard-reconstruction) provides the complete call sequence, and [reconstruction architecture](architecture/reconstruction.md) explains the solver.
6. Write the resulting image and motion model, then call `finish_run()` after the chosen output step. See [outputs](usage/outputs.md) for the run layout and saved files.

## Usage

- [Configuration](usage/configuration.md)
- [Input formats](usage/input_formats.md)
- [Sampling](usage/sampling.md)
- [Motion](usage/motion.md)
- [Outputs](usage/outputs.md)
- [Tests and simulation validation](usage/tests.md)

Installation and the shortest path to a first reconstruction remain in the [Readme](../Readme.md).

Check [pipeline architecture](pipelines.md) for the reconstruction data flow and the available preimplemented pipelines.

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


