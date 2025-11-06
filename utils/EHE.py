import torch
from utils.fftnc import ifftnc, fftnc
from utils.affinemotion import translate, translate_cplximg
from utils.createArtifacts import create_mask


def EH(xx, t_n, iterations, masks, sigmas, image_shape):
    
    # device information
    device = xx.device
    # xx: (Nx, Ny, Nz, Nc)    
    grid_size = (image_shape[0], image_shape[1])
    xx = xx.reshape(image_shape)

    EHxx = torch.zeros(image_shape, device=device, dtype=torch.complex64)

    for i in range(iterations):
        mask = create_mask(masks[i], grid_size, device=device)[:,:,None,None]
        masked_data = xx * mask

        fft_data = ifftnc(masked_data, dims=(-4, -3, -2)) 
        added_coil_sensitivity = fft_data * sigmas.conj()  # Apply coil sensitivity

        EHxx_coil = torch.zeros(image_shape, device=device, dtype=torch.complex64)
        for j in range(image_shape[-1]):

            EHxx_coil[:,:,:,j] = translate(added_coil_sensitivity[:,:,:,j], [-t_n[i, 0], -t_n[i, 1], -t_n[i, 2]])
        EHxx = EHxx + EHxx_coil

    EHxx = torch.sum(EHxx, dim=-1)  # Sum over the last dimension
    return EHxx.flatten()  



def E(xx, t_n, iterations, masks, sigmas, image_shape):
    
    # device information
    device = xx.device

    # xx: (Nx*Ny*Nz)
    grid_size = (image_shape[0], image_shape[1])

    xx = xx.reshape(image_shape[:3])

    Exx = torch.zeros(image_shape, device=device, dtype=torch.complex64)  

    for i in range(iterations):

        t_xx = translate_cplximg(xx, [t_n[i, 0], t_n[i, 1], t_n[i, 2]])

        added_coil_sensitivity = t_xx.unsqueeze(-1) * sigmas  

        kspace_data = fftnc(added_coil_sensitivity, dims=(-4, -3, -2))  # FFT transform
        mask = create_mask(masks[i], grid_size, device=device)[:,:,None,None]
        Exx = Exx + kspace_data * mask
    return Exx.flatten(start_dim=0, end_dim=2)

def EHE(xx, t_n, iterations, masks, sigmas, image_shape):
    return EH(E(xx, t_n, iterations, masks, sigmas=sigmas, image_shape=image_shape), t_n, iterations, masks, sigmas=sigmas, image_shape=image_shape)