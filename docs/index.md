# Documentation

## What is GRICS-torch?
This is a PyTorch implementation of the [GRICS algorithm](algorithm.md) for motion-corrected MRI reconstruction. This algorithm jointly estimates the image and the motion model, using physiological data surrogate and a bunch of assumtions. Please read the corresponding doc for the detailed description of the [algorithm](algorithm.md).

## How to use the software

Installation and and get started are described in the [Readme](../Readme.md).

For your specific reconstruction, follow these steps:

1. Configuration. This step loads the config files and selects the input data type. The corresponding command is [`load_config(...)`](usage/configuration/configuration.md).
2. Runime setup. This step  is needed to select compute devices and create the run state. Call `runtime = initialize_runtime(params)`.
3. Data preparation. On this step, managed with [DataPreparer class](usage/data_preparer.md), the data is created or loaded from a specified file, and processed in order to extract all necessary variables needed for the joint reconstruction. Those include the binned k-space and motion data, coil sensitivity maps. If the synthetic input data was selected, DataPreparer also handles the sampling and motion simulation.
5. Joint reconstruction. [JointReconstructor](usage/joint_reconstructor.md) is the class running the GRICS reconstruction itself. It returns motion-corrected image or volume and estimated motion model. It automatically generates some outputs. See [outputs](usage/outputs.md) for the run layout and saved files.
6. Optional post-processing or dicomization (see [post-processing](usage/postprocessing.md)).

See also:

- [Configuration](usage/configuration/configuration.md)
    - [Input formats](usage/configuration/input_formats.md)
    - [Sampling](usage/configuration/sampling.md)
    - [Motion](usage/configuration/motion.md)
- [Using DataPreparer](usage/data_preparer.md)
- [Using JointReconstructor](usage/joint_reconstructor.md)
- [Postprocessing and export](usage/postprocessing.md)

- [Outputs](usage/outputs.md)
- [Tests and simulation validation](usage/tests.md)

## Repository layout

- `src/preprocessing/`: data loading, sampling simulation, motion simulation, and motion binning
- `src/reconstruction/`: joint reconstructor, encoding and motion operators, and the CG solver
- `src/runtime/`: configuration loading and runtime initialization
- `src/utils/`: plotting, diagnostics, and notebook display helpers
- `config/`: TOML configuration root
- `pipelines/`: executable end-to-end reconstruction pipelines for real acquisitions
- `docs/`: conventions, architecture, and usage guides

## How the software is organized

The software provides a high-level interface for users who prefer not to work directly with the code, while keeping its components modular and maintainable. All components follow the shared [data and coordinate conventions](conventions.md). The following documents describe component responsibilities and data flow:

- [Configuration](architecture/configuration.md)
- [Input data loading](architecture/input_data_loading.md)
- [Data preparation](architecture/data_preparation.md)
- [Reconstruction](architecture/reconstruction.md)
- [Pipelines](architecture/pipelines.md) for the reconstruction data flow and the available preimplemented pipelines, including the breast-MRI workflows used in [1, 2].

## Implementation details

The following documents describe the main implementation details:

- [JointReconstructor: updates and regularization normalization](implementation/joint_reconstructor.md)
- [ConjugateGradientSolver: regularization, convergence, and stagnation](implementation/conjugate_gradient_solver.md)
- [MotionSimulator: synthetic acquisition and temporal models](implementation/motion_simulator.md)
- [MotionBinner: state selection and representative signals](implementation/motion_binner.md)
- [CoilSensitivityCalculator: calibration, smoothing, and normalization](implementation/coil_sensitivity_calculator.md)
- [MotionOperator: interpolation, boundaries, and adjoint](implementation/motion_operator.md)
- [EncodingOperator: shared-image encoding and faster normal operations](implementation/encoding_operator.md)
- [MotionPerturbationSimulator: Jacobian approximation and boundary gradients](implementation/motion_perturbation_simulator.md)
- [PseudoGaussSeidelMotionPreconditioner: local factors and CPU execution](implementation/motion_preconditioner.md)

## References

[1] Isaieva, K., Meullenet, C., Vuissoz, P. A., Fauvel, M., Nohava, L., Laistler, E., ... & Odille, F. (2023). Feasibility of online non-rigid motion correction for high-resolution supine breast MRI. Magnetic Resonance in Medicine, 90(5), 2130-2143.

[2] Isaieva, K., Weber, N., Nohava, L., Fischer, B., Megel, A., Henrot, P., ... & Odille, F. (2025). Optimal sensor selection for motion-corrected supine breast MRI with a wearable coil. IEEE Transactions on Biomedical Engineering, 72(7), 2261-2268.


