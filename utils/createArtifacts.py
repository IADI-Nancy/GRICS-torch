import numpy 
import torch
import torch.nn.functional as F

from utils.affinemotion import translate, translate_cplximg
from utils.fftnc import ifftnc, fftnc

def gaussian3d(shape, alpha = 2, center=None, device=None, dtype=torch.float32, normalize=True):
    """
    This function creates a matrix with a gaussian distriubtion inside.
    shape: (D, H, W)
    alpha: a bigger number makes the gauß function smaller (formular: length of side/alpha)
    center: None -> random float center inside volume, or (cz, cy, cx)
    """
    D, H, W = shape
    device = device or "cpu"

    z = torch.arange(D, device=device, dtype=dtype)
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    if center is None:
        cz = torch.rand((), device=device) * (D - 1)
        cy = torch.rand((), device=device) * (H - 1)
        cx = torch.rand((), device=device) * (W - 1)
    else:
        cz, cy, cx = [torch.as_tensor(v, device=device, dtype=dtype) for v in center]

    sz = D/alpha
    sy = H/alpha
    sx = W/alpha

    g = torch.exp(-0.5 * (((zz - cz) / sz) ** 2 +
                          ((yy - cy) / sy) ** 2 +
                          ((xx - cx) / sx) ** 2))

    if normalize:
        g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    return g

def create_coil_sensitivity(shape, alpha = 2, device = None):
    #stacking two gaussian distributions as complex value to simulate the coils
    g_real = gaussian3d(shape=shape, alpha=alpha, device = device)
    g_imag = gaussian3d(shape=shape, alpha=alpha, device = device)
    return torch.complex(g_real, g_imag) #should always be complex64 -> because of float32 input

# can be ignored 
def create_coil_data(img, shape, alpha = 2, num_coils = 7, device = None):
    img = torch.from_numpy(img).to(device)
    img = torch.complex(img, torch.zeros_like(img))
    D, H, W = shape
    coil_data = torch.zeros((D, H, W, num_coils), device=device, dtype = torch.complex64)
    for i in range(num_coils):
        coil_data[:,:,:,i] = img * create_coil_sensitivity(shape=shape, alpha=alpha, device=device)
    return coil_data

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

# main function 
def create_coil_data_with_artifacts(img, sigma = 0.05,  alpha = 2,   num_coils = 7, device = "cpu", iterations = 20):
    '''
    This functions needs a real numpy image as input and simulates sensitivity maps for the image as well as motion. 
    It returns the coil data as well as the ground truth of the motion. 
    img -> numpy image 3D
    sigma -> bigger --> more movement
    alpha -> gaußfunction --> smaller 
    num_coils -> number of coils of the simulated data
    device -> gpu/cpu
    iterations -> movement iterations during phase 

    return coil data as 4D torch array, ground truth [iterations, 3] as torch array
    '''

    img = torch.from_numpy(img).to(device).to(torch.complex64)
    
    shape = img.shape
    nX, nY, nZ = img.shape
    nCha = num_coils
    coil_data_artifact = torch.zeros((nX, nY, nZ, nCha), device=device, dtype = torch.complex64)
    
    # Generate random masks for movements 
    grid_size = (nX, nY)  
    locs, _ = torch.sort(torch.randperm(grid_size[0])[:iterations-1])

    masks = gen_masks(iterations-1, locs, grid_size, device=device) # Generate masks for each movement

    # beta controls the draw back to the original position
    beta = 0.05

    t_n = torch.zeros((iterations, 3), device=device)  
    offset = torch.zeros(3, device=device) 

    # Generate semi random translations Ornstein-Uhlenbeck process beta(pull to middle)/sigma(bigger Movement)
    for i in range(iterations):
        noise = torch.randn(3, device=device) * sigma
        offset = offset - beta * offset + noise
        t_n[i] = offset

    k_space = fftnc(img, dims = (-3,-2,-1))
    img_artifact = torch.zeros_like(img)

    # We translate the image in kspace and multiply it with a binary mask
    for i in range(iterations):
        img_moved = translate_cplximg(img_cplx = k_space, t = t_n[i], kspace = True)
        
        mask = create_mask(masks[i], grid_size, device=device)
        img_artifact = img_artifact + img_moved * mask.unsqueeze(-1)

    img_aritfact = ifftnc(X = img_artifact, dims = (-3,-2,-1))
    
    for i in range(nCha):
        coil_data_artifact[:,:,:,i] = img_artifact * create_coil_sensitivity(shape = shape, alpha = alpha, device = device) 

    return coil_data_artifact, t_n

def randomTranslation_3D(
    img: torch.Tensor,
    alpha: float = 0.5,
    sigma: float = 1.0,
    device: str = "cpu",
    iterations: int = 5,
    seed: int = 42,
    is_2D: bool = False,
    ):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not img.ndim == 4:
        return ValueError("Image dimensions are not 4.")
    img = img.to(device)
    
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