import torch
import sigpy as sp


def initialize_runtime(params, print_gpu_info=False):
    try:
        import cupy as cp
        cupy_ok = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        cupy_ok = False

    sp_device = sp.Device(0) if cupy_ok else sp.Device(-1)
    t_device = torch.device("cuda:0" if cupy_ok and torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if print_gpu_info:
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Total GPU memory: {total_mem:.2f} GB")

    if params.debug_flag:
        torch.manual_seed(params.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(params.seed)
            torch.cuda.manual_seed_all(params.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return sp_device, t_device
