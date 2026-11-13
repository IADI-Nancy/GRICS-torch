"""ESPIRiT on the GPU: cuSOLVER's SVD limit and the Gram-matrix replacement."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from src.preprocessing.CoilSensitivityCalculator import CoilSensitivityCalculator  # noqa: E402


def test_gesvd_limit_is_detected():
    if not cp.cuda.is_available():
        pytest.skip("no GPU")
    assert CoilSensitivityCalculator._gesvd_workspace_fits(cp, 50653, 6912)        # 4 coils, widths 48/12 in 3D
    assert not CoilSensitivityCalculator._gesvd_workspace_fits(cp, 50653, 12096)   # 7 coils


def test_gram_svd_matches_the_dominant_subspace():
    if not cp.cuda.is_available():
        pytest.skip("no GPU")
    rng = np.random.default_rng(0)
    A = cp.asarray((rng.standard_normal((400, 60)) + 1j * rng.standard_normal((400, 60))).astype(np.complex64))
    _, s_ref, vh_ref = cp.linalg.svd(A, full_matrices=False)
    _, s, vh = CoilSensitivityCalculator._gram_svd(A)
    assert float(cp.abs(s - s_ref).max() / s_ref[0]) < 1e-4
    keep = int((s_ref > 0.02 * s_ref[0]).sum())
    P_ref = vh_ref[:keep].conj().T @ vh_ref[:keep]      # projector onto the kept subspace (phase-free)
    P = vh[:keep].conj().T @ vh[:keep]
    assert float(cp.abs(P - P_ref).max()) < 1e-3
