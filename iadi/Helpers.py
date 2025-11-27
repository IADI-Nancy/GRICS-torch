   
import torch
import torch.nn.functional as F

def resize_img_2D(img, new_size):
    """
    Bilinear resize for complex or real images.
    Supports:
        - img: [H, W] (real or complex)
        - img: [H, W, C] (real or complex)
    """
    is_complex = img.is_complex()

    # ---------- Helper: interpolate real/imag ----------
    def interp_part(x):
        """Interpolate real-valued tensor of shape [H,W] or [H,W,C]."""
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            out = F.interpolate(x, size=new_size, mode="bilinear", align_corners=False)
            return out[0, 0]

        elif x.ndim == 3:
            C = x.shape[2]
            out_list = []
            for c in range(C):
                xc = x[:, :, c].unsqueeze(0).unsqueeze(0)
                rc = F.interpolate(xc, size=new_size, mode="bilinear", align_corners=False)
                out_list.append(rc[0, 0])
            return torch.stack(out_list, dim=2)

        else:
            raise ValueError(f"Unexpected shape {x.shape}")

    # ---------- Real tensor case ----------
    if not is_complex:
        return interp_part(img)

    # ---------- Complex case ----------
    real = interp_part(img.real)
    imag = interp_part(img.imag)
    return torch.complex(real, imag)


def from_espirit_to_grics_dims(data):
    """ Nx, Ny, Nz, Ncha  <-  Ncha, Nx, Ny, Nz """
    return data.permute(1,2,3,0).contiguous()  # width, height, slices, coils

def from_grics_to_espirit_dims(data):
    """ Nx, Ny, Nz, Ncha  ->  Ncha, Nx, Ny, Nz """
    return data.permute(3,0,1,2).contiguous()  # width, height, slices, coils