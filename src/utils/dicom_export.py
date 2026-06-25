from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import ismrmrd
import numpy as np
import torch



def write_reconstruction_dicom(
    image: Any,
    output_path: str | Path,
    raw_data: Any | None = None,
    *,
    ismrmrd_file: str | Path | None = None,
    slice_index: int | None = None,
    series_description: str = "GRICS reconstruction",
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    images_in_acquisition: int | None = None,
    series_number: int | None = None,
) -> Path:
    """
    Write a single-frame DICOM file for a reconstructed MR image.

    Patient, study, scanner, sequence, and geometry fields are copied from the
    raw-data ISMRMRD header when available. ``raw_data`` may be a DataLoader
    instance produced by this repository; otherwise pass ``ismrmrd_file``.

    ``image`` may be a NumPy array or Torch tensor with shape ``[Nx, Ny]``,
    ``[Nex, Nx, Ny]``, ``[Nx, Ny, 1]``, or ``[Nex, Nx, Ny, 1]``.
    Multi-Nex images are averaged before export. Complex images are exported
    as magnitude.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = _read_ismrmrd_header(_resolve_ismrmrd_file(raw_data, ismrmrd_file))
    geometry = _require_slice_geometry(raw_data, slice_index)
    pixels_float = _prepare_single_frame_image(image, geometry=geometry)
    pixel_array, rescale_slope = _scale_to_uint16(pixels_float)

    ds = _base_dataset(output_path)
    _copy_raw_header_fields(
        ds,
        header,
        series_description,
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        frame_of_reference_uid=frame_of_reference_uid,
        series_number=series_number,
    )
    _copy_geometry_fields(ds, header, pixel_array.shape, slice_index, images_in_acquisition, raw_data, geometry)
    _set_pixel_fields(ds, pixel_array, rescale_slope)
    _add_minimal_mr_fields(ds, header)

    ds.save_as(output_path, write_like_original=False)
    return output_path



def _ensure_pydicom_available() -> None:
    try:
        import pydicom  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "write_reconstruction_dicom requires pydicom. Install the repo pip "
            "requirements or run `pip install pydicom`."
        ) from exc


def _resolve_ismrmrd_file(raw_data: Any | None, ismrmrd_file: str | Path | None) -> Path:
    if ismrmrd_file is not None:
        return Path(ismrmrd_file)

    if raw_data is not None:
        for attr in ("source_ismrmrd_file", "ismrmrd_file"):
            value = getattr(raw_data, attr, None)
            if value:
                return Path(value)
        filenames = getattr(raw_data, "rawdata_filenames", None)
        if filenames:
            return Path(filenames[0])

    raise ValueError("Pass either raw_data with a source_ismrmrd_file or ismrmrd_file=...")


def _read_ismrmrd_header(path: Path):
    dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=False)
    try:
        return ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    finally:
        dset.close()


def _prepare_single_frame_image(image: Any, geometry: Any | None = None) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 4:
        if arr.shape[-1] != 1:
            raise ValueError(f"Expected a single slice in the last dimension, got shape {arr.shape}.")
        arr = arr.mean(axis=0)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image after preparation, got shape {arr.shape}.")

    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    # Magic flip to match Siemens DICOM export orientation. This is not documented anywhere.
    arr = np.flip(arr, axis=(0, 1))
    arr = np.asarray(arr, dtype=np.float64)
    return np.nan_to_num(arr, copy=False)


def _dicom_matrix_shape(header: Any) -> tuple[int, int] | None:
    enc = _first_present(header, "encoding", index=0)
    recon = _get(enc, "reconSpace")
    matrix = _get(recon, "matrixSize")
    x = _get(matrix, "x")
    y = _get(matrix, "y")
    if x and y:
        return int(x), int(y)
    return None


def _scale_to_uint16(image: np.ndarray) -> tuple[np.ndarray, float]:
    image = image - float(np.min(image))
    vmax = float(np.max(image))
    if vmax <= 0.0:
        return np.zeros(image.shape, dtype=np.uint16), 1.0
    scale = 65535.0 / vmax
    return np.rint(image * scale).astype(np.uint16), 1.0 / scale


def _base_dataset(output_path: Path):
    _ensure_pydicom_available()
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    now = datetime.now()
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S.%f")
    ds.ImageType = ["DERIVED", "PRIMARY", "M", "ND"]
    ds.ConversionType = "WSD"
    return ds


def _copy_raw_header_fields(
    ds: Any,
    header: Any,
    series_description: str,
    *,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    series_number: int | None = None,
) -> None:
    from pydicom.uid import generate_uid

    subject = _first_present(header, "subjectInformation")
    study = _first_present(header, "studyInformation")
    measurement = _first_present(header, "measurementInformation")
    system = _first_present(header, "acquisitionSystemInformation")

    ds.PatientName = _get(subject, "patientName", "Anonymous")
    ds.PatientID = _get(subject, "patientID", "UNKNOWN")
    _set_if_present(ds, "PatientBirthDate", _dicom_date(_get(subject, "patientBirthdate")))
    _set_if_present(ds, "PatientSex", _patient_sex(_get(subject, "patientGender")))

    ds.StudyInstanceUID = study_instance_uid or generate_uid()
    ds.SeriesInstanceUID = series_instance_uid or generate_uid()
    ds.FrameOfReferenceUID = frame_of_reference_uid or generate_uid()
    if series_number is not None:
        ds.SeriesNumber = int(series_number)
    _set_if_present(ds, "StudyID", _get(study, "studyID"))
    _set_if_present(ds, "AccessionNumber", _get(study, "accessionNumber"))
    _set_if_present(ds, "ReferringPhysicianName", _get(study, "referringPhysicianName"))
    _set_if_present(ds, "StudyDate", _dicom_date(_get(study, "studyDate")))
    _set_if_present(ds, "StudyTime", _dicom_time(_get(study, "studyTime")))

    protocol_name = _get(measurement, "protocolName")
    raw_series_description = _get(measurement, "seriesDescription", "GRICS reconstruction")
    ds.SeriesDescription = series_description or raw_series_description
    _set_if_present(ds, "ProtocolName", protocol_name)
    _set_if_present(ds, "SeriesDate", _dicom_date(_get(measurement, "seriesDate")))
    _set_if_present(ds, "SeriesTime", _dicom_time(_get(measurement, "seriesTime")))
    _set_if_present(ds, "Manufacturer", _get(system, "systemVendor"))
    _set_if_present(ds, "ManufacturerModelName", _get(system, "systemModel"))
    _set_if_present(ds, "InstitutionName", _get(system, "institutionName"))
    _set_if_present(ds, "StationName", _get(system, "stationName"))
    _set_if_present(ds, "DeviceSerialNumber", _get(system, "systemSerialNumber"))


def _copy_geometry_fields(
    ds: Any,
    header: Any,
    shape: tuple[int, int],
    slice_index: int | None,
    images_in_acquisition: int | None,
    raw_data: Any | None,
    geometry: Any | None = None,
) -> None:
    enc = _first_present(header, "encoding", index=0)
    recon = _get(enc, "reconSpace")
    fov = _get(recon, "fieldOfView_mm")
    matrix = _get(recon, "matrixSize")

    rows, cols = int(shape[0]), int(shape[1])
    ds.Rows = rows
    ds.Columns = cols
    if images_in_acquisition is not None:
        ds.ImagesInAcquisition = int(images_in_acquisition)

    fov_y = _get(fov, "y")
    fov_x = _get(fov, "x")
    if fov_y is None or fov_x is None:
        raise ValueError("DICOM export requires reconSpace fieldOfView_mm x and y.")
    ds.PixelSpacing = [float(fov_x) / float(rows), float(fov_y) / float(cols)]

    if geometry is None:
        geometry = _require_slice_geometry(raw_data, slice_index)
    _set_instance_number(ds, raw_data, geometry, slice_index)
    _set_slice_geometry_fields(ds, header, fov, raw_data, geometry)
    _set_scanner_geometry(ds, geometry, rows, cols)


def _set_instance_number(ds: Any, raw_data: Any | None, geometry: Any, slice_index: int | None) -> None:
    geometry_by_slice = getattr(raw_data, "_source_slice_geometry", None) if raw_data is not None else None
    current_location = _slice_location_from_geometry(geometry)
    if geometry_by_slice and current_location is not None:
        locations = []
        for other_geometry in geometry_by_slice.values():
            location = _slice_location_from_geometry(other_geometry)
            if location is not None:
                locations.append(location)
        if locations:
            ordered_locations = sorted(locations, reverse=True)
            ds.InstanceNumber = min(
                range(len(ordered_locations)),
                key=lambda idx: abs(ordered_locations[idx] - current_location),
            ) + 1
            return
    ds.InstanceNumber = 1 if slice_index is None else int(slice_index) + 1


def _set_slice_geometry_fields(ds: Any, header: Any, fov: Any, raw_data: Any | None, geometry: Any) -> None:
    fov_z = _get(fov, "z")
    if fov_z is not None:
        ds.SliceThickness = float(fov_z)

    patient_position = geometry.get("patient_position")
    if patient_position is None:
        measurement = _first_present(header, "measurementInformation")
        raw_patient_position = _get(measurement, "patientPosition")
        if raw_patient_position is not None:
            patient_position = str(raw_patient_position).split(".")[-1]
    _set_if_present(ds, "PatientPosition", patient_position)

    geometry_by_slice = getattr(raw_data, "_source_slice_geometry", None) if raw_data is not None else None
    spacing = _slice_spacing_from_geometry(geometry_by_slice)
    if spacing is not None:
        ds.SpacingBetweenSlices = float(spacing)


def _slice_location_from_geometry(geometry: Any) -> float | None:
    position = _vector(geometry.get("position"), 3)
    slice_dir = _unit_vector(geometry.get("slice_dir"), 3)
    if position is None or slice_dir is None:
        return None
    return float(np.dot(position, slice_dir))


def _slice_spacing_from_geometry(geometry_by_slice: Any | None) -> float | None:
    if not geometry_by_slice or len(geometry_by_slice) < 2:
        return None

    slice_locations = []
    for geometry in geometry_by_slice.values():
        location = _slice_location_from_geometry(geometry)
        if location is not None:
            slice_locations.append(location)

    if len(slice_locations) < 2:
        return None
    slice_locations = np.sort(np.asarray(slice_locations, dtype=np.float64))
    diffs = np.diff(slice_locations)
    diffs = np.abs(diffs[diffs != 0.0])
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _require_slice_geometry(raw_data: Any | None, slice_index: int | None) -> Any:
    if raw_data is None:
        raise ValueError("DICOM export requires raw_data with Siemens slice geometry.")
    if slice_index is None:
        raise ValueError("DICOM export requires slice_index so slice geometry can be selected.")
    geometry_by_slice = getattr(raw_data, "_source_slice_geometry", None)
    if not geometry_by_slice:
        raise ValueError("DICOM export requires raw slice geometry; none was loaded from the raw data.")
    geometry = geometry_by_slice.get(int(slice_index))
    if geometry is None:
        raise ValueError(f"DICOM export missing raw geometry for slice {int(slice_index)}.")
    return geometry


def _set_scanner_geometry(ds: Any, geometry: Any, rows: int, cols: int) -> None:
    position = _vector(geometry.get("position"), 3)
    read_dir = _unit_vector(geometry.get("read_dir"), 3)
    phase_dir = _unit_vector(geometry.get("phase_dir"), 3)
    slice_dir = _unit_vector(geometry.get("slice_dir"), 3)
    if position is None:
        raise ValueError("DICOM export requires a valid 3D ImagePositionPatient source position.")
    if read_dir is None or phase_dir is None:
        raise ValueError("DICOM export requires valid read and phase directions from the raw data.")
    if slice_dir is None:
        raise ValueError("DICOM export requires a valid slice direction from the raw data.")

    row_spacing = _first_float(_get(ds, "PixelSpacing"), 0)
    col_spacing = _first_float(_get(ds, "PixelSpacing"), 1)
    row_cosine = phase_dir
    column_cosine = read_dir
    ipp = position - 0.5 * float(cols - 1) * col_spacing * row_cosine - 0.5 * float(rows - 1) * row_spacing * column_cosine

    ds.ImageOrientationPatient = [float(v) for v in np.concatenate([row_cosine, column_cosine])]
    ds.ImagePositionPatient = [float(v) for v in ipp]
    ds.SliceLocation = float(np.dot(position, slice_dir))


def _vector(value: Any, length: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != length:
        return None
    return arr.reshape(length)


def _unit_vector(value: Any, length: int) -> np.ndarray | None:
    arr = _vector(value, length)
    if arr is None:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return None
    return arr / norm


def _first_float(value: Any, index: int) -> float:
    try:
        return float(value[index])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("DICOM export requires a valid two-value PixelSpacing.") from exc


def _set_pixel_fields(ds: Any, pixel_array: np.ndarray, rescale_slope: float) -> None:
    window_width = int(pixel_array.max()) if int(pixel_array.max()) > 0 else 1
    ds.WindowCenter = float(window_width) / 2.0
    ds.WindowWidth = float(window_width)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SmallestImagePixelValue = int(pixel_array.min())
    ds.LargestImagePixelValue = int(pixel_array.max())
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.RescaleType = "US"
    ds.PixelData = pixel_array.astype("<u2", copy=False).tobytes()
    ds["PixelData"].VR = "OW"


def _mr_acquisition_type(header: Any) -> str:
    enc = _first_present(header, "encoding", index=0)
    limits = _get(enc, "encodingLimits")
    recon = _get(enc, "reconSpace")
    encoded = _get(enc, "encodedSpace")
    recon_matrix = _get(recon, "matrixSize")
    encoded_matrix = _get(encoded, "matrixSize")

    n_slices = _encoding_limit_size(_get(limits, "slice"))
    n_kz = _encoding_limit_size(_get(limits, "kspace_encoding_step_2"))
    recon_z = int(_get(recon_matrix, "z", 1) or 1)
    encoded_z = int(_get(encoded_matrix, "z", 1) or 1)

    if n_kz > 1 or recon_z > 1 or encoded_z > 1:
        return "3D"
    if n_slices > 1 or n_kz == 1:
        return "2D"
    raise ValueError("Unable to determine MR acquisition type from ISMRMRD encoding limits.")


def _encoding_limit_size(limit_obj: Any) -> int:
    if limit_obj is None:
        return 1
    maximum = _get(limit_obj, "maximum")
    if maximum is None:
        return 1
    return int(maximum) + 1

def _add_minimal_mr_fields(ds: Any, header: Any) -> None:
    seq = _first_present(header, "sequenceParameters")
    _set_if_present(ds, "RepetitionTime", _first_number(_get(seq, "TR")))
    _set_if_present(ds, "EchoTime", _first_number(_get(seq, "TE")))
    _set_if_present(ds, "InversionTime", _first_number(_get(seq, "TI")))
    _set_if_present(ds, "FlipAngle", _first_number(_get(seq, "flipAngle_deg")))
    ds.ScanningSequence = "RM"
    ds.SequenceVariant = "NONE"
    ds.ScanOptions = ""
    ds.MRAcquisitionType = _mr_acquisition_type(header)
    from pydicom.sequence import Sequence

    ds.ReferencedImageSequence = Sequence([])


def _first_present(obj: Any, attr: str, index: int | None = None) -> Any:
    value = _get(obj, attr)
    if index is not None and isinstance(value, (list, tuple)) and len(value) > index:
        return value[index]
    return value


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    value = getattr(obj, attr, default)
    return default if value is None else value


def _set_if_present(ds: Any, name: str, value: Any) -> None:
    if value not in (None, ""):
        setattr(ds, name, value)


def _first_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dicom_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).replace("-", "")
    return text[:8] if len(text) >= 8 else None


def _dicom_time(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).replace(":", "").replace(".", "")
    return text[:6] if len(text) >= 6 else None


def _patient_sex(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    if text.startswith("M"):
        return "M"
    if text.startswith("F"):
        return "F"
    if text.startswith("O"):
        return "O"
    return None
