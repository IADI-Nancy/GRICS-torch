import torch 

from utils.fftnc import fftnc, ifftnc
from utils.espiritmaps import calc_espirit_maps, to_espirit_dims, from_espirit_dims
from utils.createArtifacts import randomTranslation_3D, randomRotation_3D
from utils.EHE import E, EH, EHE
from utils.conjugate_gradient import cg, nonlinear_cg
import utils.configs as cfg

def solve_known_motion_artifacts(
        kspace: torch.Tensor,
        model: int,  # 0: translation, 1: rotation
        solver: int, # 0: conjugate gradient, 1: adam
        iterations: int,
        sigma: float, # maximum motion amplitude 
        seed: int,
        sp_device
):
    """ Solve for known motion artifacts using conjugate gradient method"""
    
    with torch.no_grad():
        # calculate espirit maps and convert the dimensions from 
        espirit_maps = from_espirit_dims(calc_espirit_maps(kspace, acs = cfg.get_acs_size(), kernel_width = cfg.get_kernel_size(), sp_device = sp_device))
        kspace = from_espirit_dims(kspace)

        # Information about the data
        image_shape = kspace.shape 
        device = kspace.device

        # calculate the ground truth image
        img_cplx = ifftnc(kspace, dims = (-4, -3, -2))
        p_true = torch.sum(img_cplx*espirit_maps.conj(), dim=-1)  # Ground truth image
        print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
        
        # Simulate motion artifacts
        if model == 0:
            # translation model
            kspace_corrupted, t_n, masks = randomTranslation_3D(
                img_cplx,
                alpha=cfg.get_alpha(),
                sigma=sigma,
                iterations=iterations,
                seed=seed,
                is_2D=(image_shape[2]==1),
            )
        elif model == 1:
            # rotation model
            kspace_corrupted, t_n, masks = randomRotation_3D(
                img_cplx,
                alpha=cfg.get_alpha(),
                sigma=sigma,
                iterations=iterations,
                seed=seed,
                is_2D=(image_shape[2]==1),
            )
        else:
            raise ValueError("Unknown motion model", model)
        
        #free up as much gpu space as possible
        p_true.to("cpu")
        del img_cplx, kspace, espirit_maps
        torch.cuda.empty_cache()

        print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
        # recalculating the corrupted espirit maps shifting dims to calculate them
        espirit_maps_corrupted = from_espirit_dims(calc_espirit_maps(to_espirit_dims(kspace_corrupted), acs = cfg.get_acs_size(), kernel_width=cfg.get_kernel_size(), sp_device = sp_device))

        # calculating the corrupted image
        p_corrupted = torch.sum(ifftnc(kspace_corrupted, dims = (-4, -3, -2))*espirit_maps_corrupted.conj(), dim=-1)
        p_corrupted.to("cpu")
        torch.cuda.empty_cache()
    # solver 
    if solver == 0:
        with torch.no_grad():
            # Conjugate Gradient Solver
            # hermitian system 
            b = EH(kspace_corrupted, t_n = t_n, iterations = iterations, masks= masks, sigmas = espirit_maps_corrupted, image_shape = image_shape, model=model)
            A = EHE(torch.ones_like(b), t_n = t_n, iterations = iterations, masks= masks, sigmas = espirit_maps_corrupted, image_shape = image_shape, model=model)
            # regularisation parameter
            del kspace_corrupted
            torch.cuda.empty_cache()
            lambda_scaled = (cfg.get_lambda() * torch.norm(b, p=2))
            x_rec, _ = cg(
                A=A,
                b=b,
                x0=torch.zeros_like(b),
                max_iter=cfg.get_iterations_cg(),
                tol=cfg.get_tolerance_cg(),
                regularisation=lambda_scaled
                ) 
    elif solver == 1:
        # non-linear CG
        with torch.no_grad():
            b = EH(kspace_corrupted, t_n = t_n, iterations = iterations, masks= masks, sigmas = espirit_maps_corrupted, image_shape = image_shape, model=model)
            A = EHE(torch.ones_like(b), t_n = t_n, iterations = iterations, masks= masks, sigmas = espirit_maps_corrupted, image_shape = image_shape, model=model)
            del kspace_corrupted
            torch.cuda.empty_cache()
            x_rec, _ = nonlinear_cg(
                D=A,
                b=b,
                x0=torch.zeros_like(b),
                lam=cfg.get_lambda(),
                max_iter=cfg.get_iterations_cg(),
                tol=cfg.get_tolerance_cg(),
                beta_=cfg.get_beta()
            )
    elif solver == 2:
        # Adam solver
        E_ = E(torch.ones(image_shape[:3], dtype = torch.complex64, device = device), t_n = t_n, iterations = iterations, masks= masks, sigmas = espirit_maps_corrupted, image_shape = image_shape, model=model)
        x_rec_dims = image_shape[:-1] + (1,)
        x_rec = torch.zeros(x_rec_dims, dtype = torch.complex64, device = device, requires_grad=True)
        E_ = ifftnc(E_.reshape(image_shape), dims = (-4, -3, -2))
        image_corrupted = ifftnc(kspace_corrupted, dims = (-4, -3, -2))
        
        L1 = torch.nn.L1Loss()

        optimizer = torch.optim.AdamW(
            [x_rec],
            lr=cfg.get_learning_rate()
        )

        for epoch in range(cfg.get_iterations_adam()):
            optimizer.zero_grad()
            pred = E_ * x_rec
            loss = L1(pred, image_corrupted)
            loss.backward()
            optimizer.step()

    else:
        raise ValueError("Unknown solver", solver)
    
    # return reconstructed image, ground truth image, corrupted image
    return p_true, p_corrupted, x_rec.reshape(image_shape[:-1])
    
    








