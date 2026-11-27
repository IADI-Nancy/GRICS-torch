"""Choose FFT coil batches from live GPU memory without changing the data model."""

import torch


class CoilBatchPlanner:
    """Reserve headroom for FFT temporaries and keep each operator's batch cap stable.

    Mandatory GPU data/work tensors must fit independently. This policy bounds
    coil intermediates and can place full k-space outputs on CPU; it does not
    offload source data or cap motion caches.
    Map precision conversion is limited to the requested coil slice.
    CPU operators keep all coils in one batch.
    """

    workspace_factor = 8

    def __init__(self, smaps, compute_dtype, label):
        self._smaps = smaps
        self.compute_dtype = compute_dtype
        self.device = smaps.device
        self.n_coils = smaps.shape[0]
        self.n_voxels = smaps[0].numel()
        self.element_size = torch.empty((), dtype=compute_dtype).element_size()
        self.label = label
        self._batch_size = self.n_coils
        self._reported_size = None
        self._reported_map_conversion = False

    def coil_maps(self, start, stop, *, conjugate=False):
        """Convert only the active coil slice; never retain a full-bank copy."""
        if self.device.type == 'cuda' and self._smaps.dtype != self.compute_dtype and not self._reported_map_conversion:
            print(f'[{self.label}] Coil maps stay in source storage; {self._smaps.dtype} -> {self.compute_dtype} conversion is limited to the current FFT coil batch.', flush=True)
            self._reported_map_conversion = True
        maps = self._smaps[start:stop].to(self.compute_dtype)
        return maps.conj() if conjugate else maps

    def kspace_output_device(self, nex, nsamples):
        """Keep a full k-space buffer off CUDA when it would crowd FFT workspace."""
        budget = self._memory_budget()
        size = self.n_coils * nex * nsamples * 16  # Canonical complex128 storage.
        if budget is not None and size > budget // 2:
            if not self.__dict__.get('_reported_host_kspace', False):
                print(f'[{self.label}] Full k-space buffer requires {size / 2**30:.2f} GiB; temporary GPU budget is {budget / 2**30:.2f} GiB. Storing this buffer on CPU and transferring coil slices; FFTs remain on GPU.', flush=True)
                self._reported_host_kspace = True
            return torch.device('cpu')
        return self.device

    def kspace_samples(self, data, start, stop, nex, indices):
        """Read selected samples from CPU or GPU storage at compute precision."""
        return data[start:stop, nex, indices.to(data.device)].to(device=self.device, dtype=self.compute_dtype)

    def _memory_budget(self):
        if self.device.type != 'cuda':
            return None
        free, total = torch.cuda.mem_get_info(self.device)
        # Driver-free memory excludes unused blocks already held by PyTorch.
        reusable = max(0, torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device))
        reserve = min(2**30, total // 20)
        return max(0, free + reusable - reserve) // 2

    def state_budget(self, requested_bytes):
        """Keep existing state batching inside the same live FFT workspace budget."""
        budget = self._memory_budget()
        return requested_bytes if budget is None else min(requested_bytes, budget // self.workspace_factor)

    def batches(self, n_states=1):
        """Return coil ranges, reducing the cap as other GPU allocations grow."""
        budget = self._memory_budget()
        if budget is not None:
            per_coil = self.n_voxels * self.element_size * n_states * self.workspace_factor
            if budget < per_coil:
                raise MemoryError(f'{self.label}: insufficient GPU workspace for one FFT coil ({per_coil / 2**20:.1f} MiB estimated, {budget / 2**20:.1f} MiB budget after headroom). Data and motion caches must fit independently of coil batching.')
            self._batch_size = min(self._batch_size, max(1, budget // per_coil))
            if self._reported_size != self._batch_size:
                print(f'[{self.label}] Automatic FFT coil batch: {self._batch_size}/{self.n_coils}; {n_states} state(s), {budget / 2**30:.2f} GiB temporary-memory budget.', flush=True)
                self._reported_size = self._batch_size
        return [(start, min(start + self._batch_size, self.n_coils)) for start in range(0, self.n_coils, self._batch_size)]
