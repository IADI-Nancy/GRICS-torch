"""Generate synthetic acquisitions from images or Shepp-Logan phantoms."""

import math
import os

import numpy as np
import torch
from skimage import io as skio
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from src.preprocessing.SamplingSimulator import SamplingSimulator
from src.utils.fftnc import fftnc


class SyntheticDataGenerator:
    """Build synthetic k-space, coil maps, and acquisition indices."""

    def __init__(self, params, t_device):
        self.params = params
        self.t_device = t_device

    @staticmethod
    def _normalize_real_image(arr):
        arr = np.asarray(arr)
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[-1] in {3, 4}:
            # RGB(A) -> grayscale luminance; ignore alpha.
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D image after conversion, got shape {arr.shape}.")
        arr = arr.astype(np.float64, copy=True)
        arr -= np.min(arr)
        denom = np.max(arr)
        if denom > 0:
            arr /= denom
        return arr

    def _create_synthetic_coil_maps(self, Nx, Ny, Ncoils, Nz=1, random_phase=True):
        X, Y = torch.meshgrid(
            torch.arange(1, Nx + 1, device=self.t_device, dtype=torch.float64),
            torch.arange(1, Ny + 1, device=self.t_device, dtype=torch.float64),
            indexing="ij")

        sigma_xy = max(Nx, Ny) / 4.0
        grid_size = math.ceil(math.sqrt(Ncoils))
        xs = torch.linspace(Nx / 4.0, 3.0 * Nx / 4.0, grid_size, device=self.t_device)
        ys = torch.linspace(Ny / 4.0, 3.0 * Ny / 4.0, grid_size, device=self.t_device)
        centers = [(x.item(), y.item()) for y in ys for x in xs][:Ncoils]

        if int(Nz) > 1:
            Z = torch.arange(1, Nz + 1, device=self.t_device, dtype=torch.float64)
            Z = Z.view(1, 1, Nz)
            sigma_z = max(float(Nz) / 2.5, 1.0)
            z0 = (Nz + 1) / 2.0

        smaps = torch.empty((Ncoils, Nx, Ny, Nz), dtype=torch.cdouble, device=self.t_device)
        for c, (x0, y0) in enumerate(centers):
            profile_xy = torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma_xy ** 2))
            if int(Nz) > 1:
                profile_z = torch.exp(-((Z - z0) ** 2) / (2.0 * sigma_z ** 2))
                profile = profile_xy.unsqueeze(-1) * profile_z
            else:
                profile = profile_xy.unsqueeze(-1)
            phase = (
                torch.exp(1j * 2.0 * math.pi * torch.rand(1, device=self.t_device))
                if random_phase else 1.0
            )
            smaps[c] = profile * phase

        self.smaps_generated = smaps
        return smaps

    def _build_synthetic_kspace_from_reference_image(self, image, Ncoils=None, random_phase=True):
        image = image.to(self.t_device, dtype=torch.float64)
        if image.ndim == 2:
            reference = image.unsqueeze(-1)
        elif image.ndim == 3:
            reference = image
        else:
            raise ValueError(
                f"Expected a 2D or 3D reference image, got shape {tuple(image.shape)}."
            )

        Nx, Ny, Nz = reference.shape
        if Ncoils is None:
            Ncoils = int(self.params.Ncoils_input)

        smaps = self._create_synthetic_coil_maps(Nx, Ny, Ncoils, Nz=Nz, random_phase=random_phase)
        ref = reference.unsqueeze(0)  # (1, Nx, Ny, Nz)
        coil_imgs = ref * smaps  # (Ncoils, Nx, Ny, Nz)
        coil_imgs = coil_imgs.unsqueeze(1).expand(-1, self.params.Nex, -1, -1, -1).contiguous()

        self.kspace = fftnc(coil_imgs, dims=(-3, -2, -1))
        self.Ncha, _, self.Nx, self.Ny, self.Nz = self.kspace.shape

        samplingSimulator = SamplingSimulator(self.Ny, self.params, self.t_device)
        (self.ky_idx, self.nex_idx, self.ky_per_motion_state, self.kz_idx, self.kz_per_motion_state) =\
            samplingSimulator._build_phase_encode_indices_and_nex(Nz=self.Nz)

    def _apply_resize_factor(self, img_np):
        factor = float(self.params.image_resize_factor)
        if factor <= 0:
            raise ValueError("image_resize_factor must be > 0.")
        if factor == 1.0:
            return img_np
        h, w = img_np.shape
        new_h = max(1, int(round(h * factor)))
        new_w = max(1, int(round(w * factor)))
        return resize(img_np, (new_h, new_w), anti_aliasing=True, preserve_range=True)

    def load_image(self, path_to_image):
        ext = os.path.splitext(path_to_image)[1].lower()
        if ext == ".npy":
            img_np = np.load(path_to_image)
        elif ext == ".npz":
            npz = np.load(path_to_image)
            keys = list(npz.keys())
            if not keys:
                raise ValueError(f"No arrays found in NPZ file: {path_to_image}")
            img_np = npz[keys[0]]
        else:
            img_np = skio.imread(path_to_image)

        img_np = self._normalize_real_image(img_np)
        img_np = self._apply_resize_factor(img_np)
        img_t = torch.from_numpy(img_np).to(self.t_device, dtype=torch.float64)
        self._build_synthetic_kspace_from_reference_image(img_t)
        return self


    def _build_shepp_logan_2d(self, N, fill_fraction):
        if not math.isfinite(fill_fraction) or not 0 < fill_fraction <= 1:
            raise ValueError("SheppLoganFillFraction must be finite and in (0, 1].")
        phantom_native = shepp_logan_phantom()
        h, w = phantom_native.shape
        # Add margins by padding the native phantom before the final resize.
        # This preserves the phantom shape definition without pre-shrinking it.
        canvas_h = max(h, int(round(h / fill_fraction)))
        canvas_w = max(w, int(round(w / fill_fraction)))
        pad_top = (canvas_h - h) // 2
        pad_bottom = canvas_h - h - pad_top
        pad_left = (canvas_w - w) // 2
        pad_right = canvas_w - w - pad_left
        phantom_padded = np.pad(
            phantom_native,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
        phantom_np_2d = resize(
            phantom_padded,
            (N, N),
            anti_aliasing=True,
        )
        return phantom_np_2d

    def _build_shepp_logan_3d(self, N, Nz, fill_fraction):
        # 3D modified Shepp-Logan approximation using a sum of ellipsoids.
        if not math.isfinite(fill_fraction) or not 0 < fill_fraction <= 1:
            raise ValueError("SheppLoganFillFraction must be finite and in (0, 1].")
        x = np.linspace(-1.0, 1.0, N, dtype=np.float64) / fill_fraction
        y = np.linspace(-1.0, 1.0, N, dtype=np.float64) / fill_fraction
        z = np.linspace(-1.0, 1.0, Nz, dtype=np.float64) / fill_fraction
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        phantom = np.zeros((N, N, Nz), dtype=np.float64)

        # (A, a, b, c, x0, y0, z0, phi_deg), with in-plane rotation phi.
        ellipsoids = [
            (1.0,   0.69,   0.92,   0.90,   0.0,    0.0,    0.0,    0.0),
            (-0.8,  0.6624, 0.874,  0.88,   0.0,   -0.0184, 0.0,    0.0),
            (-0.2,  0.11,   0.31,   0.22,   0.22,   0.0,    0.0,  -18.0),
            (-0.2,  0.16,   0.41,   0.28,  -0.22,   0.0,    0.0,   18.0),
            (0.1,   0.21,   0.25,   0.41,   0.0,    0.35,  -0.15,   0.0),
            (0.1,   0.046,  0.046,  0.05,   0.0,    0.10,   0.25,   0.0),
            (0.1,   0.046,  0.046,  0.05,   0.0,   -0.10,   0.25,   0.0),
            (0.1,   0.046,  0.023,  0.05,  -0.08,  -0.605,  0.0,    0.0),
            (0.1,   0.023,  0.023,  0.02,   0.0,   -0.606,  0.0,    0.0),
            (0.1,   0.023,  0.046,  0.02,   0.06,  -0.605,  0.0,    0.0),
        ]

        for A, a, b, c, x0, y0, z0, phi_deg in ellipsoids:
            phi = np.deg2rad(phi_deg)
            x_shift = X - x0
            y_shift = Y - y0
            z_shift = Z - z0
            x_rot = x_shift * np.cos(phi) + y_shift * np.sin(phi)
            y_rot = -x_shift * np.sin(phi) + y_shift * np.cos(phi)
            inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 + (z_shift / c) ** 2 <= 1.0
            phantom[inside] += A

        # Match in-plane orientation of the canonical 2D Shepp-Logan (major axis vertical).
        phantom = np.swapaxes(phantom, 0, 1)

        # Keep the expected non-negative intensity profile.
        phantom = np.clip(phantom, 0.0, None)
        max_val = float(phantom.max())
        if max_val > 0:
            phantom /= max_val
        return phantom

    def generate_shepp_logan(self, N=128, Ncoils=4, Nz=1, random_phase=True):
        fill_fraction = float(self.params.SheppLoganFillFraction)
        if int(Nz) > 1:
            phantom_np = self._build_shepp_logan_3d(N, Nz, fill_fraction)
            phantom = torch.tensor(phantom_np, dtype=torch.float64, device=self.t_device)
        else:
            phantom_np_2d = self._build_shepp_logan_2d(N, fill_fraction)
            phantom_2d = torch.tensor(phantom_np_2d, dtype=torch.float64, device=self.t_device)
            phantom = phantom_2d.unsqueeze(-1).expand(N, N, Nz)  # (Nx, Ny, Nz)
        self.phantom_generated = phantom
        self._build_synthetic_kspace_from_reference_image(phantom, Ncoils=Ncoils, random_phase=random_phase)
        return self
