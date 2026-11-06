"""Share one loaded acquisition across CPU slice workers."""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import torch

LOADED_DATA = None
RECONSTRUCT_SLICE = None


def _initialize_worker(input_data, reconstruct_slice):
    global LOADED_DATA, RECONSTRUCT_SLICE
    LOADED_DATA, RECONSTRUCT_SLICE = input_data, reconstruct_slice
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        print(f"[workers] Keeping the existing inter-op thread limit: {error}", flush=True)


def _reconstruct_slice(slice_idx):
    result = RECONSTRUCT_SLICE(slice_idx, LOADED_DATA)
    # NumPy transport avoids PyTorch shared-memory lifetime issues after workers exit.
    for key in ("image", "native_image", "motion"):
        result[key] = result[key].numpy()
    return result


def reconstruct_slices_in_parallel(input_data, slice_indices, reconstruct_slice, max_workers=None):
    """Return CPU reconstruction results in slice order."""
    if not slice_indices:
        raise ValueError("At least one slice must be selected for reconstruction.")
    if max_workers is not None and (type(max_workers) is not int or max_workers < 1):
        raise ValueError("max_workers must be a positive integer or None.")
    if input_data.params.runtime_device != "cpu":
        raise ValueError("Slice workers only support CPU execution.")
    # Use one worker if the operating system cannot report its CPU count.
    workers = min(max_workers or (os.cpu_count() or 1), len(slice_indices))
    if workers == 1:
        return [reconstruct_slice(index, input_data) for index in slice_indices]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"),
                             initializer=_initialize_worker, initargs=(input_data, reconstruct_slice)) as executor:
        results = list(executor.map(_reconstruct_slice, slice_indices))
    for result in results:
        for key in ("image", "native_image", "motion"):
            result[key] = torch.from_numpy(result[key])
    return results
