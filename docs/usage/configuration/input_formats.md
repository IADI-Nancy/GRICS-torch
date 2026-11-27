# Input formats

The `data_type` selected in `load_config(...)` controls how input data is built or loaded. For 2D real data, `slice_idx` selects the slice/partition to reconstruct. Pass it only to `prepare_reconstructor_inputs(slice_idx=...)`, after `load_input_data()`; constructors configure the source without selecting a slice. It may be omitted only when the source contains exactly one slice; multi-slice sources require an explicit value before preparation. For synthetic data and all 3D data, do not provide `slice_idx`; input preparation raises an error if it is set.

## Synthetic data types
### `shepp-logan`

`load_config(...)` automatically selects `config/synthetic_data/shepp_logan_2d.toml` or `shepp_logan_3d.toml` from `reconstruction_dimension` in the reconstruction TOML.

In addition to `reconstruction_config`, pass:

- a sampling simulation config file
- a motion simulation config file (otherwise there is nothing to correct)

### `from_image`

Loaded from a 2D image file and converted to synthetic multi-coil k-space using generated coil maps. Supported inputs include common image formats (e.g. PNG/JPEG/TIFF) and NumPy arrays (`.npy`, `.npz`).

`load_config(...)` automatically loads `config/synthetic_data/from_image.toml`.

In addition to `reconstruction_config`, pass:

- a sampling simulation config file
- a motion simulation config file

## Real data types
When `save_debug_plots=true` and sampling is `from-data`, every real-world input mode uses the same source-independent acquisition-order filename in the reconstruction’s `preprocessing/` folder: `ky_order_acquisition_slice{slice_idx}.png`.

### `preprocessed-real`

For 2D multi-slice data, load a preprocessed HDF5 file with datasets:
- `kspace`: shape `(Ncoils, Nex, Nx, Ny, Nslices)`, complex (`complex64`/`complex128`)
- `motion_data`: shape `(Nslices, Nlines)` for one channel or `(Nslices, Nlines, Nchannels)` for multiple channels, real (`float32`/`float64`) - motion data associated with each k-space line (navigator/respiratory bellow indications, etc.)
- `idx_ky`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)
- `idx_kz`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)
- `idx_nex`: shape `(Nslices, Nlines)`, integer (`int32`/`int64`)

For 3D data, `kspace` has shape `(Ncoils, Nex, Nx, Ny, Nz)` and `motion_data` has shape `(Nreadout, Nchannels)`. Acquisition indices contain one entry per readout in full acquisition order; the reader-generated layout is `(1, Nreadout)`. Index values are zero-based.

No synthetic sampling is needed with `from-data`: acquisition order and motion signals come from the file. Additional motion simulation can still be applied. If simulated sampling and motion are selected instead, only `kspace` is required; the recorded acquisition indices and physiological signals are not used.

### `ismrmrd-navigators` and `siemens-navigators`

These modes derive physiological motion channels from navigators embedded in the MRI acquisition. Pass `filename` as a path or a dictionary containing only `ismrmrd_file` or `siemens_raw_file`, respectively. Siemens conversion selects its single measurement for VB files or the last measurement declared by a VD/VE file. No external physiological recording is needed. `load_config(...)` automatically loads `config/physiological_data/navigators.toml`.

The `navigator_type` setting selects the acquisition-specific extraction implementation in `ISMRMRDReader`; currently only `"mosar"` is supported, and unknown types are rejected. The MOSAR implementation reads navigator signals from the second contrast (zero-based set counter 1). `navigator_imaging_set` selects the imaging contrast, 0 or 1 (default 1). The reader identifies the three central readouts at the start of each set/repetition block, and removes them from the selected contrast's imaging and calibration k-space. Other repeated central runs inside an imaging block are rejected as ambiguous. The strategy supports one encoded slice/slab and requires at least two navigator triplets. It does not fall back to navigation flags: those flags can also mark Siemens RT-feedback/VOP records, which are excluded. Triplet ordinals 0, 1, and 2 are acquisition-order labels, not verified anatomical directions.

`NavigatorEigenmodeCalculator`, under `src/preprocessing/real_data/physiological_data/`, removes declared readout oversampling from navigator samples, Fourier-transforms them into profiles, replaces the configured startup profiles, and registers each coil/ordinal series to its final profile. The calculator consumes grouped navigator records independently of their acquisition type and stacks all selected displacement fields for one uncentered SVD. `navigator_eigenmodes` controls how many temporal modes become physiological channels; it must not exceed the measured matrix's numerical rank. `navigator_directions` selects a nonempty subset of triplet ordinals; `[0]`, `[2]`, and `[0, 1, 2]` reproduce the first-only, third-only, and combined variants. Registration remains non-rigid with spatial regularization and no temporal penalty. Translation fitting and interleaved-set experiments remain in the draft study scripts under `validation_studies/navigator_motion_draft/`.

The timestamped modes enter the same `RawDataPreparer` synchronization and acquisition-wide normalization as other physiological sources, followed by the existing motion binning and reconstruction preparation. Both MRI samples and navigator channels retain the same scanner-clock origin; a nonzero `physio_clock_drift_seconds` override is rejected. `navigator_extrapolation` explicitly selects the shared boundary policy. The default `"edge"` holds the final mode values over imaging readouts acquired after the last navigator and reports the uncovered interval; interior values use linear interpolation. `"raise"` rejects uncovered readouts, while `"autoregression"` permits at most one second of extrapolation. These channels are surrogates, not measured 3D translations or rotations. `navigator_eigenmodes` counts channels; `N_motion_states` separately controls binning.

After `load_input_data()`, `input_loader.kspace` contains the selected contrast's imaging data and `input_loader.navigators` contains the second contrast's decoded central-triplet records. `input_loader.raw_data_preparer.physiological_reader.metadata` provides navigator times, retained modes, singular values, numerical rank, and source indices. Normalized per-readout signals are exposed through the existing motion-data interface. `prepare_reconstructor_inputs()` now supports these modes like other physiological data types. Preprocessing caches include the extraction/calculation settings, synchronized data, navigator records, and channel metadata.

For low-level inspection, `ISMRMRDReader(path, read_navigators=True).read_data()` still returns flagged acquisition records under `navigators` when no navigator type is supplied. `read_navigator_data()` reads those flagged records independently of imaging-grid support. Neither flag-based inspection method establishes that the records are anatomical motion navigators. The production MOSAR path explicitly supplies `navigator_type="mosar"` and `navigator_set=1` instead; `imaging_set=0` can select the first contrast while retaining second-set navigator signals.

### `ismrmrd-physio_array`
- MRI raw data in ISMRMRD format (`ismrmrd_file`) is loaded using `ISMRMRDReader`. `RawDataPreparer` combines it with the physiological data and converts the inputs to the arrays used by the `preprocessed-real` mode.
- External physiological data is provided as two `.npy` files: `physio_timestamps_file`, of shape `(Nsensors, Nsamples, 1)`, containing timestamps in seconds, and `physio_values_file`, of shape `(Nsensors, Nsamples, Ntracks)`, containing real data values. Tracks of the same sensor share timestamps. Each sensor's last timestamp must correspond to the end of the MRI sequence; timestamps must be strictly increasing.
- If the data is already synchronized with MRI readouts, set every timestamp to `-1`. In this case, each sensor/track must contain exactly one value for every retained MRI imaging readout in full-acquisition order, before slice selection; interpolation is skipped and clock correction must be zero. Do not mix channels whose timestamps are all `-1` with timestamped channels. An isolated `-1` in a strictly increasing timestamp sequence is an ordinary time value.

Pass `filename` to `DataLoader` as a dictionary with the three keys above, or as `(ismrmrd_file, physio_timestamps_file, physio_values_file)`. All sensor/track pairs are kept as separate motion channels, ordered by sensor and then track, without filtering or normalization.

### `siemens-physio_array`
Uses the same physiological data format together with Siemens raw scanner data (`siemens_raw_file`). The loader first converts the Siemens raw file to ISMRMRD using the `siemens_to_ismrmrd` executable, then reads the result with `ISMRMRDReader`. Replace `ismrmrd_file` with `siemens_raw_file` in the dictionary or tuple above.

### `ismrmrd-physio_text` and `siemens-physio_text`

Use the corresponding MRI data format together with a whitespace-separated physiological text file (`physio_file`). Each nonempty, non-comment row is one sample from one sensor:

```text
SENSOR TIMESTAMP VALUE1 [VALUE2 ...]
```

The first line may be a header whose first two fields are `SENSOR TIMESTAMP`; lines may include comments after `#`. `SENSOR` is a nonnegative integer identifier. `TIMESTAMP` is in seconds. `VALUE1`, `VALUE2`, and subsequent value columns are that sensor's tracks. Every data row must have the same positive number of value columns, and timestamped rows for each sensor must be in strictly increasing timestamp order; sensors may be interleaved and may have different sample counts or sampling rates. Each sensor’s final timestamp must correspond to the end of the MRI sequence, with clock correction applied after end alignment.

Use `-1` for every timestamp when values are already synchronized to MRI readouts. Then every track must provide one value for every retained full-acquisition MRI imaging readout, before slice selection; interpolation is skipped and clock correction must be zero. Do not mix channels whose timestamps are all `-1` with timestamped channels. An isolated `-1` in a strictly increasing timestamp sequence is an ordinary time value.

Pass `filename` as `(ismrmrd_file, physio_file)` or `(siemens_raw_file, physio_file)`, or as a dictionary with the corresponding keys. Text channels are ordered by numeric sensor ID and then value-column order.

### `ismrmrd-saec` and `siemens-saec`
Uses the corresponding MRI data format together with a physiological data file in SAEC [1, 2] format (`saec_file`). `load_config(...)` automatically loads `config/physiological_data/saec.toml` for `ismrmrd-saec` and `siemens-saec` data. Select the sensor type and filters in `saec.toml`.

SAEC processing depends on `rawdata_sensor_type`:

- `BELT`: selects the belt track with the larger standard deviation, applies a first-order zero-phase Butterworth low-pass filter using `saec_belt_lowpass_cutoff_hz`, and removes quadratic drift. The drift is fitted to a copy clipped to the mean plus or minus two standard deviations, then subtracted from the unclipped filtered signal.
- `1MARMOT`: for every MARMOT sensor, each accelerometer track is low-pass filtered using `saec_marmot_lowpass_cutoff_hz` and high-pass filtered using `saec_marmot_highpass_cutoff_hz`. Tracks identified as displaced, constant, non-finite, or otherwise invalid are rejected, then the valid track with the largest standard deviation is selected across all sensors.
- `ALL_MARMOTS` (also accepted as `ALL_MARMOTs`): applies the same configured filtering and displacement checks and selects the highest-variance valid track from each usable sensor.

Timestamped SAEC channels are aligned to the full MRI sequence before slice selection and interpolated onto MRI readout times. SAEC timestamps are referenced to the Siemens stop trigger. The filters are applied before this interpolation; readouts outside the SAEC recording use the nearest endpoint value.

### `ismrmrd-polaris` and `siemens-polaris`
Uses the corresponding MRI data format together with a single-tool NDI ToolBox `.tsv` export from a Polaris Vega infrared camera tracker (`polaris_file`). `load_config(...)` automatically loads `config/physiological_data/polaris.toml` for `ismrmrd-polaris` and `siemens-polaris` data. Select the tracks, filter, and clock shift in `polaris.toml`. The final timestamp must correspond to the end of the MRI sequence; clock correction is applied after end alignment.

Polaris reads the XYZ tool-position channels and applies a first-order zero-phase Butterworth low-pass filter using `polaris_lowpass_cutoff_hz`. The sampling rate is estimated from the recording duration; at least seven samples are required, and the rate must exceed twice the cutoff. The filtered signals are then interpolated onto full-sequence MRI readout times. With `polaris_channel_mode = "all"`, Tx, Ty, and Tz are retained. With `"largest-amplitude"`, only the axis with the largest peak-to-peak range is retained (ties are resolved in Tx, Ty, Tz order). Shared per-channel normalization follows selection.

### Post-processing and synchronization
Text inputs automatically load `config/physiological_data/physio_text.toml`; array inputs load `config/physiological_data/physio_array.toml`. Each file owns its clock-shift value independently. Edit the corresponding file or use `overrides={"physio_clock_drift_seconds": ...}` for a run-specific value.

Generic `physio_text` and `physio_array` inputs are not filtered. Their channels are preserved in sensor/track order through timestamp alignment and interpolation, then standardized by the shared per-channel normalization. If all timestamps are `-1`, values are treated as already synchronized and must contain one sample for every full-acquisition MRI readout; interpolation is skipped and clock correction must be zero.

Physiological clock correction (Polaris, `physio_text`, `physio_array`): set `physio_clock_drift_seconds` in `config/physiological_data/polaris.toml` for Polaris, `config/physiological_data/physio_text.toml` for text inputs, or `config/physiological_data/physio_array.toml` for array inputs (default `0.0`), or pass `overrides={"physio_clock_drift_seconds": -0.25}`. Positive shifts samples later; negative shifts earlier, after sequence-end alignment. Already-synchronized (`-1`) inputs reject any nonzero clock correction. Missing coverage at either edge is extrapolated with $\hat{x}[n] = c + \sum_{k=1}^{p} a_k x[n-k]$, fitted per channel (up to 20 lags over the nearest 10 seconds; reversed history at the start). Any required extension greater than 1 second raises an error. SAEC is unchanged.

## Loading performance

ISMRMRD loading filters acquisition headers before decoding sample arrays. Readout oversampling is removed from acquired imaging and calibration lines in batches targeting 32 MiB of complex128 FFT input, before mapping them onto the expanded centered grid; navigator samples remain as acquired. Reader logs report the selected/total acquisition counts and elapsed decoding/gridding time. A caller can reuse verified navigator registration and temporal modes between contrasts in memory; synchronization remains independent for each contrast. This reuse does not create persistent files or change `cache_preprocessed_data` or `remove_temporary_data_after_run`.

## References

[1] Isaieva, K., Fauvel, M., Weber, N., Vuissoz, P. A., Felblinger, J., Oster, J., & Odille, F. (2022). A hardware and software system for MRI applications requiring external device data. Magnetic Resonance in Medicine, 88(3), 1406-1418.

[2] IADI-Nancy. [wrapperHDF5](https://github.com/IADI-Nancy/wrapperHDF5).
