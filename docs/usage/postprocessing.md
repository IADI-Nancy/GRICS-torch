# Postprocessing and export

`JointReconstructor.run()` returns the native image and motion model produced by the solver. Postprocessing is an optional calling-workflow step that turns that reconstruction into the image required for review, analysis, storage, or exchange.

## Why postprocess

The solver produces an image on the grid that is convenient for reconstruction. Before the image is reviewed or exported, the pipeline prepares it for its intended use:

1. **Restore the requested field of view and matrix.** The solver grid, encoded grid, and final reconstruction grid can differ. The pipeline zero-fills k-space to the encoded grid and center-crops to the reconstruction matrix specified in the acquisition header. This places the result on the requested final grid; it does not recreate anatomy outside the acquired field of view.
2. **Optionally normalize the intensity.** MRI intensity is not an absolute physical unit, and receive-coil sensitivity can make the same tissue appear brighter in one location than another. MRI manufacturers commonly offer intensity-correction options for this reason. GRICS-torch can similarly divide the reconstruction by its prepared GRICS reference image, producing a more uniform presentation scale when this is appropriate for the workflow.
3. **Encode pixels for DICOM.** DICOM output uses an unsigned 16-bit container, with 12 bits stored for each pixel. The exporter therefore converts the final magnitude image to integer values from 0 to 4095. This DICOM scaling is required for the file representation and is separate from the optional GRICS-reference normalization.

Keep the native image whenever the final image is changed so the solver result remains available for quality assessment or another downstream transformation. The returned motion model remains on the solver grid; changing the image grid does not transform its motion coordinates.

## Standard pipeline postprocessing

The Siemens 2D and 3D pipelines apply the following sequence after `JointReconstructor.run()`:

1. Preserve the solver output as `native_image`. This is the complex image on the solver grid and remains available for quality checks or a different downstream workflow.
2. Optionally normalize the image by the prepared GRICS reference image. The pipeline divides by the reference magnitude, replacing reference values below its mean magnitude with that mean to avoid unstable amplification in low-signal regions.
3. Restore the requested image grid. The pipeline restores the header encoded matrix in Fourier space, padding a smaller solver grid or trimming the reader's known acceleration-expanded grid, then center-crops the image to the reconstruction matrix in the ISMRMRD header. If the image already has the requested shape, this step leaves it unchanged.
4. Save and export the final image. Tensor output uses the normalized and zero-filled/cropped image. DICOM export then takes its magnitude and scales it to the 12-bit stored pixel range; the motion model remains on the solver grid.

The normalization setting comes from `config/postprocessing/nonrigid_2d_breast.toml` or `config/postprocessing/nonrigid_3d_breast.toml`. The zero-fill helpers obtain their encoded and reconstruction sizes from the acquisition header. If a converted header omits a reconstruction size, they derive it from the encoded matrix and field-of-view ratio.

For a prepared `data` object, the corresponding image steps are:

```python
from pipelines.utils.postprocessing import (
    grics_reference_image_for_normalization, grics_zero_fill_shapes,
    grics_zero_fill_volume_shapes, normalize_reconstruction_by_grics_reference,
    zero_fill_loaded_reconstruction,
)
from src.runtime.runtime_config import load_postprocessing_config

config_file = f"config/postprocessing/nonrigid_{data.params.data_dimension.lower()}_breast.toml"
postprocessing = load_postprocessing_config(config_file)
image, motion = reconstructor.run()
native_image = image.detach().cpu()
if postprocessing.normalize_image_by_grics_reference:
    reference_image = grics_reference_image_for_normalization(data, image)
    image, _ = normalize_reconstruction_by_grics_reference(image, reference_image)
shape_reader = grics_zero_fill_volume_shapes if data.params.data_dimension == "3D" else grics_zero_fill_shapes
target_shape, encoded_shape = shape_reader(data.input_loader)
image = zero_fill_loaded_reconstruction(image, target_shape, encoded_shape)
```

`grics_zero_fill_shapes(...)` reads the in-plane matrix sizes for 2D; `grics_zero_fill_volume_shapes(...)` adds the partition sizes for 3D. Both validate the loaded grid against the encoded grid and the reader's acceleration-dependent expansion. A supported expanded grid is reported explicitly and restored by centered Fourier cropping before the image-space field-of-view crop; unrelated oversized grids are rejected. `zero_fill_loaded_reconstruction(...)` applies the validated shape to the final two or three spatial axes with [`zero_fill_grics_image_to_shape(...)`](../../src/utils/zero_fill.py). The original motion model stays on the solver grid. In the 2D pipeline, this sequence runs in each slice worker; in the 3D pipeline, it runs once on the full volume.

## Save the native and final results

Call `write_reconstruction(...)` after the final image has been chosen. It saves the configured tensors and records metadata about the image stage. Pass `native_image` and set `image_stage="after_postprocessing"` so the output metadata distinguishes the saved final image from the solver image.

```python
from src.runtime.output_layout import write_reconstruction

write_reconstruction(
    data.params, image, motion, native_image=native_image,
    image_stage="after_postprocessing",
)
```

When no image transformation is applied, save the image returned by `run()` and use the default `image_stage`, `"before_postprocessing"`. The [output guide](outputs.md) describes the output files and saving flags.

## Export

Export follows postprocessing because exported pixels should represent the final chosen image. The Siemens 2D pipeline writes one DICOM per prepared slice with [`write_reconstruction_dicom(...)`](../../src/utils/dicom_export.py); the 3D pipeline uses [`write_volume_dicoms(...)`](../../src/utils/dicom_export.py) to expand the slab center into slice positions. Both convert the postprocessed complex image to magnitude, shift its minimum to zero, and scale its maximum to a value no greater than 4095. They store these values as `uint16` pixels with 12 meaningful bits (`BitsAllocated = 16`, `BitsStored = 12`). The volume exporter derives z spacing from the encoded z field of view and matrix size, preserving spacing even if z was zero-filled or cropped. The DICOM geometry and pixel matrix must describe the final image grid.

The Siemens pipelines provide concrete normalization, zero-filling, and DICOM-export examples. See the [2D T2 pipeline](../architecture/pipelines.md#2d-t2-pipeline), the [3D low-resolution pipeline](../architecture/pipelines.md#3d-low-resolution-pipeline), and the [output guide](outputs.md).

## MOSAR inversion images and combination

Run `python pipelines/mosar.py acquisition.dat --output-root runs/mosar` with a Siemens `.dat` input. The pipeline accepts only Siemens files and converts the acquisition once internally for both contrasts. Generated conversions are removed after use when `remove_temporary_data_after_run = true` (the default). `--device cpu|gpu` selects computation (default `gpu`); `--reconstruction-config` selects a 3D rigid or non-rigid reconstruction file (default `config/reconstruction/nonrigid_3d_mosar.toml`). `--dicom-header-dir` optionally supplies public metadata. `--dicom-series-number` selects the first of three consecutive series numbers, default 1001. Python callers can pass normal reconstruction overrides to `run_pipeline(...)`; navigator eigenmode settings come from `config/physiological_data/navigators.toml`.

The pipeline reconstructs the two inversion contrasts independently, rather than averaging them. INV2 is reconstructed first so its coil sensitivity maps can be reused for INV1, preserving their relative complex phase. Both use second-set navigator modes on the shared scanner clock. The pipeline explicitly replaces per-contrast physiological standardization with INV2 readout means and standard deviations for both contrasts, preserving the same motion-signal zero. It retains the configured navigator endpoint policy, including holding values before the first or after the last navigator when `navigator_extrapolation = "edge"`. Shared navigator signals assume that the measured motion surrogate also describes the intervening first-set readouts; there is no joint motion fit across contrasts.

Both complex volumes are zero-filled/cropped to the header reconstruction grid before combination. Independent reference-image intensity normalization is omitted to preserve their relative amplitude. The combination is `real(conj(INV1) * INV2) / (abs(INV1)**2 + abs(INV2)**2)`, yielding values from −0.5 to 0.5; voxels with exactly zero denominator are explicitly set to zero and counted in the log. This follows the commented expression in the supplied MATLAB script and the standard phase-sensitive MP2RAGE combination [1]. The script's external `mp2rage_recon` implementation was not supplied, so any additional processing inside that helper is not reproduced.

The acquisition run contains `exports/dicom/inv1/`, `exports/dicom/inv2/`, and `exports/dicom/combined/`, each with one DICOM file per reconstructed partition. The series have distinct identifiers and consecutive series numbers, with a shared study and frame of reference. INV1 and INV2 export magnitudes with fixed scaling over each complete volume. The combined series exports signed values through 12-bit unsigned pixels with `RescaleIntercept = -0.5` and `RescaleSlope = 1/4095`; a DICOM reader must apply these tags to recover the signed image. `VOILUTFunction = "LINEAR_EXACT"` supplies continuous display windowing over that range [2]. Inversion time and flip angle come from the matching sequence entry when available and are omitted for the combined image. Geometry and slice positions match across all three series.

`exports/mosar.json` records the source acquisition, shared motion normalization, combination formula, and DICOM paths/identifiers. Normal reconstruction tensors are controlled by `save_reconstruction_tensors`; when enabled, the INV2 result folder also contains `image_mp2rage_combined.pt`. The returned dictionary includes `contrasts`, `combined_image`, `dicom_series`, `manifest_file`, and `run_folder`. The two contrast reconstructions use the existing solver separately; shared motion estimation across both images is not implemented.

Use `--no-motion-correction` with `pipelines/mosar.py` to reconstruct INV1 and INV2 iteratively with motion fixed to zero at every resolution. Coil-map estimation, image regularization, CG settings, and contrast combination stay configured as usual; navigator extraction and binning still run. The three exported DICOM series include `NO_MOCO` in their descriptions, and run metadata records `motion_correction=false`. Omitting the flag enables motion correction.

## References

[1] Marques, J. P., et al. (2010). MP2RAGE, a self bias-field corrected sequence for improved segmentation and T1-mapping at high field. NeuroImage, 49(2), 1271–1281. [PubMed](https://pubmed.ncbi.nlm.nih.gov/19819338/).

[2] NEMA. [DICOM PS3.3, C.11.2: VOI LUT Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html).
