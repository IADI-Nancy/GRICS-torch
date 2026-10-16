# Input formats

The `data_type` selected in `load_config(...)` controls how input data is built or loaded.
For 2D real data, `slice_idx` selects the slice/partition to reconstruct. It may be omitted only when the source contains exactly one slice; multi-slice sources require an explicit value before running the slice pipeline.
For synthetic data and for all 3D data, do not provide `slice_idx`; the loader raises an error if it is set.

## Synthetic data types
### `shepp-logan`

Required config files:
- `config/synthetic_data/shepp_logan_2d.toml` (or `config/synthetic_data/shepp_logan_3d.toml` for 3D data)
- a sampling simulation config file
- a motion simulation config file (otherwise there is nothing to correct)

### `from_image`

Loaded from a 2D image file and converted to synthetic multi-coil k-space using generated coil maps.
Supported inputs include common image formats (e.g. PNG/JPEG/TIFF) and NumPy arrays (`.npy`, `.npz`).

Required config files:
- `config/synthetic_data/from_image.toml`
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

### `ismrmrd-physio_array`
- MRI raw data in ISMRMRD format (`ismrmrd_file`) is loaded using `ISMRMRDReader` with `config/real_data/ismrmrd_reader.toml`. `RawDataPreparer` combines it with the physiological data and converts the inputs to the arrays used by the `preprocessed-real` mode.
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

Pass `filename` as `(ismrmrd_file, physio_file)` or `(siemens_raw_file, physio_file)`, or as a dictionary with the corresponding keys. Text channels are ordered by numeric sensor ID and then value-column order. These four generic physiological modes require `ismrmrd_reader_config="config/real_data/ismrmrd_reader.toml"` in `load_config(...)`.

### `ismrmrd-saec` and `siemens-saec`
Uses the corresponding MRI data format together with physiological data file in SAEC [3, 4] format (`saec_file`). Pass `real_data_config="config/real_data/saec.toml"` and `ismrmrd_reader_config="config/real_data/ismrmrd_reader.toml"` to `load_config(...)`. Select the sensor type in `saec.toml`.

SAEC processing depends on `rawdata_sensor_type`:

- `BELT`: selects the belt track with the larger standard deviation, applies a first-order zero-phase Butterworth low-pass filter using `saec_belt_lowpass_cutoff_hz`, and removes quadratic drift. The drift is fitted to a copy clipped to the mean plus or minus two standard deviations, then subtracted from the unclipped filtered signal.
- `1MARMOT`: for every MARMOT sensor, each accelerometer track is low-pass filtered using `saec_marmot_lowpass_cutoff_hz` and high-pass filtered using `saec_marmot_highpass_cutoff_hz`. Tracks identified as displaced, constant, non-finite, or otherwise invalid are rejected, then the valid track with the largest standard deviation is selected across all sensors.
- `ALL_MARMOTS` (also accepted as `ALL_MARMOTs`): applies the same configured filtering and displacement checks and selects the highest-variance valid track from each usable sensor.

Timestamped SAEC channels are aligned to the full MRI sequence before slice selection and interpolated onto MRI readout times. SAEC timestamps are referenced to the Siemens stop trigger. The filters are applied before this interpolation; readouts outside the SAEC recording use the nearest endpoint value.

### `ismrmrd-polaris` and `siemens-polaris`
Uses the corresponding MRI data format together with a single-tool NDI ToolBox `.tsv` export from a Polaris Vega infrared camera tracker (`polaris_file`). Pass `polaris_config="config/real_data/polaris.toml"` and `ismrmrd_reader_config="config/real_data/ismrmrd_reader.toml"` to `load_config(...)`. Select the tracks and clock shift in `polaris.toml`. The final timestamp must correspond to the end of the MRI sequence; clock correction is applied after end alignment.

Polaris reads the XYZ tool-position channels and applies a first-order zero-phase Butterworth low-pass filter using `polaris_lowpass_cutoff_hz`. The sampling rate is estimated from the recording duration; at least seven samples are required, and the rate must exceed twice the cutoff. The filtered signals are then interpolated onto full-sequence MRI readout times. With `polaris_channel_mode = "all"`, Tx, Ty, and Tz are retained. With `"largest-amplitude"`, only the axis with the largest peak-to-peak range is retained (ties are resolved in Tx, Ty, Tz order). Shared per-channel normalization follows selection.

### Post-processing and synchronization
Text inputs automatically load `config/real_data/physio_text.toml`; array inputs load `config/real_data/physio_array.toml`. Each file owns its clock-shift value independently. To use another sensor configuration, pass `physio_config="path/to/physio_text.toml"` to `load_config(...)`. The clock setting belongs to the sensor configuration, not `ismrmrd_reader.toml`.

Generic `physio_text` and `physio_array` inputs are not filtered. Their channels are preserved in sensor/track order through timestamp alignment and interpolation, then standardized by the shared per-channel normalization. If all timestamps are `-1`, values are treated as already synchronized and must contain one sample for every full-acquisition MRI readout; interpolation is skipped and clock correction must be zero.

Physiological clock correction (Polaris, `physio_text`, `physio_array`): set `physio_clock_drift_seconds` in `config/real_data/polaris.toml` for Polaris, `config/real_data/physio_text.toml` for text inputs, or `config/real_data/physio_array.toml` for array inputs (default `0.0`), or pass `overrides={"physio_clock_drift_seconds": -0.25}`. Positive shifts samples later; negative shifts earlier, after sequence-end alignment. Already-synchronized (`-1`) inputs reject any nonzero clock correction. Missing coverage at either edge is extrapolated with $\hat{x}[n] = c + \sum_{k=1}^{p} a_k x[n-k]$, fitted per channel (up to 20 lags over the nearest 10 seconds; reversed history at the start). Any required extension greater than 1 second raises an error. SAEC is unchanged.

SAEC sources [3] and [4] are listed in the [project references](../../Readme.md#references).
