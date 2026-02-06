   
import torch





def from_espirit_to_grics_dims(data):
    """ Nx, Ny, Nz, Ncha  <-  Ncha, Nx, Ny, Nz """
    return data.permute(1,2,3,0).contiguous()  # width, height, slices, coils

def from_grics_to_espirit_dims(data):
    """ Nx, Ny, Nz, Ncha  ->  Ncha, Nx, Ny, Nz """
    return data.permute(3,0,1,2).contiguous()  # width, height, slices, coils

# def build_sampling_from_motion_states(ky_per_mot_state_idx, ky_idx, nex_idx, Nx, Ny, t_device):
#     sampling_idx = []

#     kx = torch.arange(Nx, device=t_device, dtype=torch.int32)

#     # -------------------------------------------------
#     # Loop over motion states
#     # -------------------------------------------------
#     for mot_state, ky_mot_state in enumerate(ky_per_mot_state_idx):
#         # ky_ms: (Nky_ms,)

#         # Determine Nex for these ky lines
#         # (they all belong to the same Nex by construction)
#         ky0 = ky_mot_state[0]

#         # Find Nex index (cheap lookup)
#         # TODO add multiple Nex support
#         # Nex_idx = 0 # (ky_idx == ky0).nonzero(as_tuple=False)[0, 0]

#         # ----- flattened sampling indices -----
#         samp = (
#             ky_mot_state.unsqueeze(0) +
#             Ny * kx.unsqueeze(1)
#         ).reshape(-1)

#         sampling_idx.append(samp)

#     return sampling_idx
def build_sampling_per_nex_per_motion(
    binned_ky_indices,  # [Nex][Nmotion]
    Nx,
    device,
):
    Nex = len(binned_ky_indices)
    Nmotion = len(binned_ky_indices[0])

    kx = torch.arange(Nx, device=device, dtype=torch.int64)

    Sampling = [
        [None for _ in range(Nmotion)]
        for _ in range(Nex)
    ]

    for nex in range(Nex):
        for ms in range(Nmotion):
            ky = binned_ky_indices[nex][ms]

            if ky.numel() == 0:
                Sampling[nex][ms] = torch.empty(
                    0, dtype=torch.int64, device=device
                )
                continue

            # Build flattened (kx, ky) sampling
            samp = (
                kx[:, None]
                + Nx * ky[None, :]
            ).reshape(-1)

            Sampling[nex][ms] = samp

    return Sampling


def kmeans_torch(x, k, n_iter=20):
    N, D = x.shape

    # Better init: random unique points
    perm = torch.randperm(N, device=x.device)
    centers = x[perm[:k]].clone()

    for _ in range(n_iter):
        dist = torch.cdist(x, centers)
        labels = dist.argmin(dim=1)

        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(dim=0)
            else:
                # 🔴 Empty cluster → reinitialize
                idx = torch.randint(0, N, (1,), device=x.device)
                centers[j] = x[idx]

    return labels, centers
