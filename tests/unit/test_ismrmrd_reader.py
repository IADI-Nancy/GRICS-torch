"""ISMRMRDReader on small synthetic files: oversampling from the header, line centring, repeated lines."""
import numpy as np
import pytest

ismrmrd = pytest.importorskip("ismrmrd")
from src.preprocessing.real_data.ISMRMRDReader import ISMRMRDReader  # noqa: E402

X = ismrmrd.xsd
N = 64


def _header(enc_x, enc_y, rec_x, rec_y, ky_limits, avg_max=0):
    def lim(mn, mx, c):
        return X.limitType(minimum=mn, maximum=mx, center=c)
    enc = X.encodingType(
        encodedSpace=X.encodingSpaceType(matrixSize=X.matrixSizeType(x=enc_x, y=enc_y, z=1),
                                         fieldOfView_mm=X.fieldOfViewMm(x=256.0 * enc_x / rec_x, y=256.0, z=5.0)),
        reconSpace=X.encodingSpaceType(matrixSize=X.matrixSizeType(x=rec_x, y=rec_y, z=1),
                                       fieldOfView_mm=X.fieldOfViewMm(x=256.0, y=256.0, z=5.0)),
        encodingLimits=X.encodingLimitsType(kspace_encoding_step_1=lim(*ky_limits), kspace_encoding_step_2=lim(0, 0, 0),
                                            slice=lim(0, 0, 0), repetition=lim(0, 0, 0), average=lim(0, avg_max, 0)),
        trajectory=X.trajectoryType.CARTESIAN)
    return X.ismrmrdHeader(experimentalConditions=X.experimentalConditionsType(H1resonanceFrequency_Hz=123000000),
                           measurementInformation=X.measurementInformationType(patientPosition=X.patientPositionType.HFS),
                           encoding=[enc])


def _write(path, hdr, lines):
    d = ismrmrd.Dataset(str(path), "dataset", create_if_needed=True)
    d.write_xml_header(X.ToXML(hdr) if hasattr(X, "ToXML") else hdr.toXML("utf-8"))
    for i, (data, ky, avg) in enumerate(lines):
        a = ismrmrd.Acquisition.from_array(np.ascontiguousarray(data.astype(np.complex64)))
        a.idx.kspace_encode_step_1 = ky
        a.idx.average = avg
        a.acquisition_time_stamp = 1000 + 4 * i
        a.center_sample = data.shape[1] // 2
        a.read_dir[:] = (1, 0, 0)
        a.phase_dir[:] = (0, 1, 0)
        a.slice_dir[:] = (0, 0, 1)
        d.append_acquisition(a)
    d.close()


def _fftc(img):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img), norm="ortho"))


def _ifftc(k):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k), norm="ortho"))


def _image():
    img = np.zeros((N, N))
    img[8:56, 20:44] = 1.0
    img[2:6, 30:34] = 3.0      # marker near the readout edge
    return img


def _read(path):
    out = ISMRMRDReader(str(path)).read_data()
    return out["kspace"].numpy()[0, 0, :, :, 0], out


def test_oversampled_file_is_cropped_to_the_recon_fov(tmp_path):
    img = _image()
    full = np.zeros((2 * N, N))
    full[N // 2:N // 2 + N] = img
    k = _fftc(full)
    _write(tmp_path / "os2.h5", _header(2 * N, N, N, N, (0, N - 1, N // 2)), [(k[:, ky][None, :], ky, 0) for ky in range(N)])
    ks, _ = _read(tmp_path / "os2.h5")
    assert ks.shape[0] == N
    assert np.abs(np.abs(_ifftc(ks)) - img).max() < 1e-6


def test_file_without_oversampling_keeps_the_full_fov(tmp_path):
    img = _image()
    k = _fftc(img)
    _write(tmp_path / "os1.h5", _header(N, N, N, N, (0, N - 1, N // 2)), [(k[:, ky][None, :], ky, 0) for ky in range(N)])
    ks, _ = _read(tmp_path / "os1.h5")
    assert ks.shape[0] == N
    assert np.abs(np.abs(_ifftc(ks)) - img).max() < 1e-6


@pytest.mark.parametrize("center", [32, 16])
def test_declared_center_is_placed_at_the_fft_center(tmp_path, center):
    # 48 acquired lines indexed 0..47 with the k-space centre declared at `center`
    img = _image()
    k = _fftc(img)
    rows = np.arange(N // 2 - center, N // 2 - center + 48)          # which full-grid lines the file holds
    lines = [(k[:, row][None, :], i, 0) for i, row in enumerate(rows)]
    _write(tmp_path / f"pf{center}.h5", _header(N, N, N, N, (0, 47, center)), lines)
    ks, out = _read(tmp_path / f"pf{center}.h5")
    assert ks.shape[1] == N
    expected = np.zeros_like(k)
    expected[:, rows] = k[:, rows]
    assert np.abs(ks - expected).max() < 1e-6
    assert sorted(out["idx_ky"].tolist()) == sorted(rows.tolist())


def test_repeated_lines_are_averaged(tmp_path, capsys):
    img = _image()
    ka, kb = _fftc(img), _fftc(0.5 * img)
    lines = [(ka[:, ky][None, :], ky, 0) for ky in range(N)] + [(kb[:, ky][None, :], ky, 1) for ky in range(N)]
    _write(tmp_path / "avg.h5", _header(N, N, N, N, (0, N - 1, N // 2), avg_max=1), lines)
    ks, _ = _read(tmp_path / "avg.h5")
    assert "acquired more than once" in capsys.readouterr().out
    assert np.abs(ks - 0.5 * (ka + kb)).max() < 1e-6
