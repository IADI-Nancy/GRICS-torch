# GRICS-torch 2.0 — Release notes

Compared with v1.0; based on current HEAD.

- **Siemens workflows:** automatic raw-data conversion to ISMRMRD; callable 2D breast T2 and 3D low-resolution pipelines; DICOM export, cropping and zero-filling utilities.
- **Expanded physiological inputs:** multiple sensor channels, SAEC MARMOT selection, Polaris tracking, and physiological text/NumPy inputs, with improved synchronization, clock correction and channel normalization.
- **Reconstruction improvements:** one shared image reconstructed jointly from multiple repetitions; optimized normal operator eliminating readout FFTs; optional non-rigid pseudo Gauss–Seidel preconditioning, calibration-weighted reconstruction and voxel-aware regularization scaling.
- **Calibration and simulation:** added Odille-spline coil sensitivity estimation alongside ESPIRiT, undersampling simulation with calibration lines, clearer per-shot/per-readout motion modes, and explicit CPU retries when GPU memory is insufficient during motion simulation.
- **Motion and solver reliability:** added k-space-energy binning and per-resolution state counts; corrected empty-state handling, multiresolution state propagation, CG stopping and selection of the best-residual iterate.
- **External integration:** new data-preparation, single-iteration and k-space prediction APIs, image/motion initialization, external image priors and caller-defined sampling masks for reconstruction and learning workflows.
- **Runtime and documentation:** reorganized preprocessing and configuration; structured run folders with manifests, resolved settings and independent log/tensor/plot controls; shared conversion/preprocessing caches and cleanup utilities; expanded algorithm, architecture, implementation and usage documentation.

**Migration:** update custom imports, input-type names, configuration paths, motion-mode names and output handling. Replace the combined output-saving flag with separate controls. Multiple repetitions now contribute to a shared reconstruction; Laplacian regularization support has been removed.
