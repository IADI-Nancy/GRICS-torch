"""Shared spatial interpolation for real and complex reconstruction tensors."""

import torch
import torch.nn.functional as F


def resize_img_xy(img, new_size):
    is_complex = img.is_complex()
    target_3d = len(new_size) == 3

    # ---------- Helper: interpolate real/imag ----------
    def interp_part(x):
        """Interpolate real-valued tensor in 2D or 3D spatial coordinates."""
        if target_3d:
            nx_new, ny_new, nz_new = new_size
            if x.ndim == 3:
                # [Nx, Ny, Nz] -> [1, 1, Nz, Nx, Ny]
                xv = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                out = F.interpolate(xv, size=(nz_new, nx_new, ny_new), mode="trilinear", align_corners=False)
                return out[0, 0].permute(1, 2, 0)  # [Nx, Ny, Nz]
            elif x.ndim == 4:
                # [C, Nx, Ny, Nz] -> [1, C, Nz, Nx, Ny]
                xv = x.permute(0, 3, 1, 2).unsqueeze(0)
                out = F.interpolate(xv, size=(nz_new, nx_new, ny_new), mode="trilinear", align_corners=False)
                return out[0].permute(0, 2, 3, 1)  # [C, Nx, Ny, Nz]
            elif x.ndim == 5:
                # [C, Nx, Ny, Nz, S] -> [C, Nx_new, Ny_new, Nz_new, S]
                c, s = x.shape[0], x.shape[-1]
                xv = x.permute(0, 4, 1, 2, 3).reshape(c * s, x.shape[1], x.shape[2], x.shape[3])
                xv = xv.permute(0, 3, 1, 2).unsqueeze(0)
                out = F.interpolate(xv, size=(nz_new, nx_new, ny_new), mode="trilinear", align_corners=False)
                return out[0].permute(0, 2, 3, 1).reshape(c, s, nx_new, ny_new, nz_new).permute(0, 2, 3, 4, 1)
            else:
                raise ValueError(f"Unexpected shape {x.shape} for 3D resize.")

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            out = F.interpolate(x, size=new_size, mode="bilinear", align_corners=False)
            return out[0, 0]

        elif x.ndim == 3:
            C = x.shape[0]
            out_list = []
            for c in range(C):
                xc = x[c].unsqueeze(0).unsqueeze(0)
                rc = F.interpolate(xc, size=new_size, mode="bilinear", align_corners=False)
                out_list.append(rc[0, 0])
            return torch.stack(out_list, dim=0)
        elif x.ndim == 4 and x.shape[-1] != 1:
            # [C, Nx, Ny, S] -> [C, Nx_new, Ny_new, S]
            c, s = x.shape[0], x.shape[-1]
            xv = x.permute(0, 3, 1, 2).reshape(c * s, x.shape[1], x.shape[2])
            out_list = []
            for idx in range(c * s):
                xc = xv[idx].unsqueeze(0).unsqueeze(0)
                rc = F.interpolate(xc, size=new_size, mode="bilinear", align_corners=False)
                out_list.append(rc[0, 0])
            return torch.stack(out_list, dim=0).reshape(c, s, new_size[0], new_size[1]).permute(0, 2, 3, 1)
        elif x.ndim == 4 and x.shape[-1] == 1:
            # 2D-with-single-z convention: [C, Nx, Ny, 1] -> [C, Nx_new, Ny_new, 1]
            C = x.shape[0]
            out_list = []
            for c in range(C):
                xc = x[c, :, :, 0].unsqueeze(0).unsqueeze(0)
                rc = F.interpolate(xc, size=new_size, mode="bilinear", align_corners=False)
                out_list.append(rc[0, 0])
            return torch.stack(out_list, dim=0).unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected shape {x.shape}")

    # ---------- Real tensor case ----------
    if not is_complex:
        return interp_part(img)

    # ---------- Complex case ----------
    real = interp_part(img.real)
    imag = interp_part(img.imag)
    return torch.complex(real, imag)
