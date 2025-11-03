import numpy 
import torch
import torch.nn.functional as F

from utils.affinemotion import translate, translate_cplximg
from utils.fftnc import ifftnc, fftnc

# helper functions for randomTranslation_3D 
def gen_masks(n_movements, locs, grid_size, device):
    masks = []
    # first element
    mask = torch.arange(0,locs[0],dtype=torch.long,device=device)
    masks.append(mask)

    for i in range(1,n_movements):
        mask = torch.arange(locs[i-1],locs[i],dtype=torch.long,device=device)
        masks.append(mask)    
    # last element
    mask = torch.arange(locs[-1],grid_size[0],dtype=torch.long,device=device)
    masks.append(mask)
    return masks

def create_mask(mask, grid_size, device):

    bin_mask = torch.zeros(grid_size, dtype=torch.bool, device=device)
    for i in mask:
        bin_mask[i, :] = True
    return bin_mask

def randomTranslation_3D(
    img: torch.Tensor,
    alpha: float = 0.5,
    sigma: float = 1.0,
    iterations: int = 5,
    seed: int = 42,
    is_2D: bool = False
    ):
    '''
    This function applies random translations to a 3D or 2D (+ coils) image.
    img : torch.Tensor 3D or 2D (nX, nY, nZ, nCha) dims with nZ = 1 for 2D images
    alpha : float, controls the draw back to the original position (Ornsteil-Uhlenbeck process)
    sigma : float, standard deviation of the random translations
    iterations : int, number of random translations to apply
    seed : int, random seed for reproducibility
    is_2D : bool, if True, only apply translations in x and y directions
    '''
    # setting up the seed and device
    device = img.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not img.ndim == 4:
        return ValueError("Image dimensions are not 4.")
    
    nX, nY, nZ, nCha = img.shape
    corrupted_img = torch.zeros_like(img, device=device)
    
    # Generate random masks for movements 
    grid_size = (nX, nY)  
    locs, _ = torch.sort(torch.randperm(grid_size[0])[:iterations-1])

    masks = gen_masks(iterations-1, locs, grid_size, device=device) # Generate masks for each movement

    # alpha controls the draw back to the original position
    t_n = torch.zeros((iterations, 3), device=device)  
    offset = torch.zeros(3, device=device) 

    # Generate semi random translations Ornstein-Uhlenbeck process 
    for i in range(iterations):
        noise = torch.rand(3, device=device) * sigma
        if is_2D:
            noise[2] = 0.0  # No movement in z-direction for 2D images
        offset = offset - alpha * offset + noise
        t_n[i] = offset
    
    # Fourier transform as if the image was in 3D
    kspace = fftnc(img)  # Centered FFT

    # Apply the translations only for x and y dimensions over all coils
    for i in range(iterations):
        kspace_movement = torch.zeros_like(kspace, device=device, dtype=kspace.dtype)
        for j in range(nCha):
            kspace_movement[:,:,:,j] = translate_cplximg(kspace[:,:,:,j], [t_n[i, 0], t_n[i, 1], t_n[i,2]])
        mask = create_mask(masks[i], grid_size, device=device)
        corrupted_img = corrupted_img + kspace_movement * mask.unsqueeze(-1).unsqueeze(-1)  # Apply the mask to the k-space
    # Inverse Fourier transform to get the corrupted image
    corrupted_img = ifftnc(corrupted_img)  # Centered IFFT

    return corrupted_img, t_n, masks