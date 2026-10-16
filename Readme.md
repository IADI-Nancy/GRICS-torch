# GRICS-torch: GRICS MRI motion-corrected reconstruction in PyTorch

This repository contains a 2D/3D MRI reconstruction pipeline with joint image-motion estimation using the GRICS algorithm [1], implemented in PyTorch with GPU support. GRICS models MRI acquisition and motion and does not require AI priors. It requires motion-related data (e.g. respiratory belt measurements, navigators, PilotTone amplitude variations, or similar signals). This implementation aims to improve understanding of the algorithm in the MRI community and support its reuse.

Please contact Karyna Isaieva (karyna [dot] isaieva [at] univ-lorraine [dot] fr) for any bug reports, questions or suggestions.

## License and citation

This project is distributed under the MIT License. See [LICENSE](LICENSE) for
the full terms.

Please cite the GRICS paper if you use this code for your research work.

```bibtex
@article{odille2008grics,
  title = {Generalized reconstruction by inversion of coupled systems (GRICS) applied to free-breathing MRI},
  author = {Odille, F. and Vuissoz, P. A. and Marie, P. Y. and Felblinger, J.},
  journal = {Magnetic Resonance in Medicine},
  volume = {60},
  number = {1},
  pages = {146--157},
  year = {2008}
}
```

## Environment setup

A Dockerfile is provided in `build/`, and the built image is available from the
[GitHub container registry](https://github.com/IADI-Nancy/GRICS-torch/pkgs/container/grics-torch).
The image uses Python 3.12.12, PyTorch 2.5.1, and CUDA 12.1. GPU execution
requires a compatible NVIDIA driver and the NVIDIA Container Toolkit on the
host; the CUDA toolkit itself is included in the image.

To run the included examples your computer should have at least 16GB of VRAM
and 64GB of RAM. An NVIDIA GPU is not required, but CPU reconstruction is much
slower. For CPU-only execution, disable the active `GPU=( --gpus ... )` line
and enable `GPU=( )` in `docker.sh`, then set `runtime_device = "cpu"` in
`config/general.toml`.

You can use `docker.sh` from the repository to start and manage the container:
```bash
./docker.sh run                 # Create the container, or start it if it already exists
./docker.sh exec-it             # Open an interactive shell as your host user
./docker.sh exec-it python --version
./docker.sh stop                # Stop the container
./docker.sh rm                  # Remove the stopped container
```

By default, the script mounts the **parent directory of this repository** at `/home/pyuser/wkdir` inside the container so this repository will be in `/home/pyuser/wkdir/GRICS-torch` (or the repository's actual directory name). For example, if you checkout this repository in the parent folder of your data and your project directories, they will be available under `/home/pyuser/wkdir/`. The script also mounts the host `~/.gitconfig`. By default it runs on GPU 0; set `IMAGE_NAME` or `EXTRA_MOUNTS` in the environment to change the image or add Docker mount arguments. `IMAGE_VERSION` selects the container-image tag; dependency versions are pinned in the files under `build/` and are not derived from that tag. Use `./docker.sh build` to build the image locally from `build/Dockerfile`.

When you `./docker.sh run` it will print a VS Code container configuration for connecting to the running container.

## Repository layout

- `src/preprocessing/`: data loading, sampling simulation, motion simulation, motion binning
- `src/reconstruction/`: joint reconstructor, encoding/motion operators, CG solver, etc.
- `src/runtime/`: config loading and runtime initialization
- `src/utils/`: plotting, diagnostics, notebook display helpers
- `config/`: TOML configuration root
- `pipelines/`: executable end-to-end reconstruction pipelines for real acquisitions; `siemens_breast_T2.py` reproduces the Gadgetron pipeline implemented in [2]
- `docs/`: pipeline overview, shared conventions, architecture, and usage guides

## Getting started

The demo notebooks expose each reconstruction stage and are the shortest path
through a first reconstruction:

- [2D non-rigid simulation from an image](demo_a_2D_nonrigid_simulated_from_image_periodic.ipynb)
- [2D non-rigid reconstruction of real free-breathing data](demo_b_2D_nonrigid_realworld_freebreathing.ipynb)
- [3D rigid Shepp-Logan simulation](demo_c_3D_rigid_simulated_shepp_interleaved.ipynb)
- [3D non-rigid Shepp-Logan simulation](demo_d_3D_nonrigid_simulated_shepp_random.ipynb)

To run a demo:

1. Start the container with `./docker.sh run`.
2. Connect VS Code to the container using the configuration printed by the
   command.
3. Open `/home/pyuser/wkdir/<repository-name>` in the container.
4. Run Demo A for the simplest end-to-end example, or choose another notebook
   from the list above.

For complete acquisitions, the callable and command-line workflows
`pipelines/siemens_breast_T2.py` and
`pipelines/siemens_breast_3d_lowres.py` coordinate the same stages with input
selection, export, and execution management. Their data flow is documented in
[pipeline architecture](docs/pipelines.md).

Simulations and some reconstruction steps use random initialization;
`seed_enabled` and `seed` in `config/general.toml` control reproducibility.
Results can vary with the seed and compute backend. Generated results are
written under `runs/` by default; see the [output guide](docs/usage/outputs.md).

## Documentation

Start with the [documentation index](docs/index.md). It links to:

- the [pipeline data flow](docs/pipelines.md)
- shared [data and coordinate conventions](docs/conventions.md)
- internal [architecture](docs/index.md#architecture)
- user-facing [configuration, input, motion, output, and API guides](docs/index.md#usage)

## Disclosure

Parts of this code and its documentation were developed with the assistance of AI tools (mostly Codex). All content has been reviewed and validated by a human.

## References

[1] Odille, F., Vuissoz, P. A., Marie, P. Y., & Felblinger, J. (2008). Generalized reconstruction by inversion of coupled systems (GRICS) applied to free‐breathing MRI. Magnetic Resonance in Medicine, 60(1), 146-157.
[2] Isaieva, K., Meullenet, C., Vuissoz, P. A., Fauvel, M., Nohava, L., Laistler, E., ... & Odille, F. (2023). Feasibility of online non‐rigid motion correction for high‐resolution supine breast MRI. Magnetic Resonance in Medicine, 90(5), 2130-2143.
[3] Isaieva, K., Fauvel, M., Weber, N., Vuissoz, P. A., Felblinger, J., Oster, J., & Odille, F. (2022). A hardware and software system for MRI applications requiring external device data. Magnetic Resonance in Medicine, 88(3), 1406-1418.
[4] https://github.com/IADI-Nancy/wrapperHDF5
