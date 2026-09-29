"""Multiplicative calibration-image prior used by GRICS++."""

import math

import torch

from src.utils.fftnc import fftnc, ifftnc


class CalibrationPriorEncodingOperator:
    """Apply an encoding operator to ``x = C p`` instead of directly to ``x``.

    Here ``C`` is the voxel-wise calibration magnitude and ``p`` is the latent
    image solved by CG. Thus ``A_C p = A(C p)`` and its adjoint is
    ``A_C^H y = C^H A^H y``. The resulting normal equation is
    ``(C^H A^H A C + lambda I) p = C^H A^H y``; the returned image is ``x = C p``.
    """

    def __init__(self, encoding_operator, calibration_image):
        self.encoding_operator = encoding_operator
        self.calibration_image = calibration_image.flatten()
        self.device = encoding_operator.device

    def forward(self, latent_image):
        # A_C p = A(C p)
        return self.encoding_operator.forward(self.calibration_image * latent_image)

    def adjoint(self, data):
        # A_C^H y = C^H A^H y.
        return self.calibration_image.conj() * self.encoding_operator.adjoint(data)

    def normal(self, latent_image):
        # A_C^H A_C p = C^H A^H A(C p)
        return self.calibration_image.conj() * self.encoding_operator.normal(
            self.calibration_image * latent_image
        )


def fourier_crop_calibration_image(image, new_size):
    """Crop ``C`` to a reconstruction level in centered Fourier space.

    The unitary-FFT normalization preserves a constant calibration image while
    retaining the same low spatial frequencies as the corresponding k-space
    resolution level.
    """
    old_size = tuple(image.shape)
    new_size = tuple(new_size)
    if len(new_size) not in (2, 3) or len(old_size) != len(new_size):
        raise ValueError("Calibration prior must have two or three spatial dimensions.")
    if any(target < 1 or target > source for source, target in zip(old_size, new_size)):
        raise ValueError(f"Invalid calibration prior crop from {old_size} to {new_size}.")
    if old_size == new_size:
        return image

    spatial_dims = tuple(range(image.ndim))
    frequency = fftnc(image, dims=spatial_dims)
    source_index = tuple(
        slice(source // 2 - target // 2, source // 2 - target // 2 + target)
        for source, target in zip(old_size, new_size)
    )
    cropped = frequency[source_index]
    return ifftnc(cropped, dims=spatial_dims) * math.sqrt(
        math.prod(new_size) / math.prod(old_size)
    )
