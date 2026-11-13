# GRICS-torch: GRICS MRI motion-corrected reconstruction in PyTorch

This repository contains a 2D/3D MRI reconstruction pipeline with joint image-motion estimation using the GRICS algorithm [1], implemented in PyTorch with GPU support. GRICS models MRI acquisition and motion and does not require AI priors. It requires motion-related data (e.g. respiratory belt measurements, navigators, PilotTone amplitude variations, or similar signals). This implementation aims to improve understanding of the algorithm in the MRI community and support its reuse.

The code currently supports only Cartesian sampling and performs best with fully sampled data. The algorithm was developed primarily for non-rigid motion correction. Rigid motion correction is also available, but has been less extensively validated; users are encouraged to assess its performance for their application as we continue to refine it.

Please contact Karyna Isaieva (karyna [dot] isaieva [at] univ-lorraine [dot] fr) for any bug reports, questions or suggestions.

## License and citation

This project is distributed under the MIT License. See [LICENSE](LICENSE) for the full terms.

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

### Docker

The simplest way to get started is to use the prepared Docker image. A Dockerfile is provided in `build/`, and the built image is available from the [GitHub container registry](https://github.com/IADI-Nancy/GRICS-torch/pkgs/container/grics-torch). The image uses CUDA 12.1. GPU execution requires a compatible NVIDIA driver and the NVIDIA Container Toolkit on the host; the CUDA toolkit itself is included in the image.

To run the included examples in native configuration your computer should have at least 16GB of VRAM and 64GB of RAM. An NVIDIA GPU is not required, but CPU reconstruction is much slower. Motion simulation automatically retries on CPU if it runs out of CUDA memory; coil-map estimation and reconstruction do not retry on CPU after a CUDA memory error. To run the complete workflow on CPU, set `runtime_device = "cpu"` in `config/general.toml` and check for device overrides in the chosen notebook or script. If there is no GPU on your workstation, also disable the active `GPU=( --gpus ... )` line and enable `GPU=( )` in `docker.sh`.

You can use `docker.sh` from the repository to start and manage the container:
```bash
./docker.sh run                 # Create the container, or start it if it already exists
./docker.sh exec-it             # Open an interactive shell as your host user
./docker.sh exec-it python --version
./docker.sh stop                # Stop the container
./docker.sh rm                  # Remove the stopped container
```

By default, the script mounts the **parent directory of this repository** at `/home/pyuser/wkdir` inside the container so this repository will be in `/home/pyuser/wkdir/GRICS-torch` (or the repository's actual directory name). For example, if you checkout this repository in the parent folder of your data and your project directories, they will be available under `/home/pyuser/wkdir/`. The script also mounts the host `~/.gitconfig`. `IMAGE_VERSION` selects the container-image tag; dependency versions are pinned in the files under `build/` and are not derived from that tag. Use `./docker.sh build` to build the image locally from `build/Dockerfile`.

After `./docker.sh run`, open the VS Code Command Palette and select **Dev Containers: Open Container Configuration File**. Paste the JSON printed by `docker.sh` into that file and save it, then connect to the running container. This makes VS Code use the container user created with the host user's UID and GID, so files created inside the mounted workspace belong to the actual host user instead of `root`.

For a comfortable VS Code setup, install [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) on the host. The configuration printed by `docker.sh` installs [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance), [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), Jupyter Keymap, and Jupyter Notebook Renderers in the container. Install [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml) in the attached container for editing configuration files.

### Without Docker

The basic installation requires Linux (Ubuntu 22.04) with Python 3.11 or 3.12. The GPU mode select CUDA 12.1 PyTorch and CUDA 12.x CuPy for GPU ESPIRiT; execution requires a compatible NVIDIA driver [2]. You can create and activate a Python environment using your preferred environment manager before installing. From the repository root in your chosen environment, first upgrade pip:

```bash
python -m pip install --upgrade "pip>=23.2.1"
```

After the upgrade succeeds, run the commands for your device:

```bash
# CPU
python -m pip install "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu &&
python -m pip install -e .
```

```bash
# NVIDIA GPU
python -m pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121 &&
python -m pip install -e ".[gpu]" &&
CUDA_WHEEL_LIBS=$(python -c 'import site; from pathlib import Path; libs = sorted({str(p) for root in site.getsitepackages() for p in Path(root).glob("nvidia/*/lib") if p.is_dir()}); assert libs, "No pip-installed NVIDIA library directories found; check the GPU installation and active Python environment"; print(":".join(libs))') &&
export LD_LIBRARY_PATH="$CUDA_WHEEL_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" &&
python -c 'from cupy_backends.cuda.libs import cusolver; print("CuPy cuSOLVER import succeeded")'
```

The commands install dependencies and the project in editable mode using `pyproject.toml`, making `src` importable outside the repository. Basic includes simulation, preprocessed HDF5 and ISMRMRD `.h5` input, reconstruction, ESPIRiT, DICOM export, and notebook kernels/widgets. For SAEC support or Siemens `.dat` conversion, see the corresponding commands in the [Dockerfile](build/Dockerfile).

The GPU sequence also exposes the CUDA libraries installed by pip to CuPy, including `libcusolver.so.11`, which ESPIRiT needs for its singular value decomposition, and verifies that cuSOLVER imports successfully. The export applies to the current shell only: repeat the `CUDA_WHEEL_LIBS` and `export` lines in every new terminal, notebook kernel or batch job that runs ESPIRiT on the GPU, or append them to the activation script of the environment (`bin/activate` for a venv). Without them, the runtime prints a notice and runs ESPIRiT on the CPU while PyTorch keeps the GPU.

Basic pip installation normally needs no additional apt packages when binary wheels are available. For native builds of the Siemens converter and its dependencies, the Ubuntu/Debian packages used by the Dockerfile are: `git`, `build-essential`, `cmake`, `pkg-config`, `libboost-all-dev`, `libpugixml-dev`, `libhdf5-dev`, `libxml2-dev`, `libxslt1-dev`, `libfftw3-dev`, `liblapacke-dev`, and `libopenblas-dev`.

### Windows users

We recommend **WSL 2 with Ubuntu and Docker Desktop**.

**1. Install WSL 2 and Ubuntu.** Run in **Windows PowerShell as administrator**:

```powershell
wsl --install -d Ubuntu
```

Follow the first-launch prompts to create your Linux username and password. Password characters do not appear while typing.

**2. Configure Docker Desktop.** Install and launch [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/). In Settings → General, enable Use the WSL 2 based engine if shown; under Resources → WSL Integration, enable Ubuntu. Apply the changes and restart Docker Desktop if prompted.

**3. Open the project in Ubuntu.** Open **Start → Ubuntu**. **IADI lab users:** configure the proxy inside Ubuntu (see the internal Wiki), before cloning. Then clone into your Linux home directory:

```bash
cd ~
git clone https://github.com/IADI-Nancy/GRICS-torch.git
cd GRICS-torch
```

Follow the [Docker instructions](#docker) to start the container and configure VS Code. **Optional:** To move Docker storage or Ubuntu from C: to another drive, see [Docker storage](docs/usage/specific_instructions_windows_users.md#move-docker-desktop-storage) and [Ubuntu storage](docs/usage/specific_instructions_windows_users.md#move-the-ubuntu-wsl-distribution).

## Getting started

The demo notebooks expose each reconstruction stage and are the shortest path through a first reconstruction:

- [2D non-rigid simulation from an image](demo_a_2D_nonrigid_simulated_from_image_periodic.ipynb)
- [2D non-rigid reconstruction of real free-breathing data](demo_b_2D_nonrigid_realworld_freebreathing.ipynb)
- [3D rigid Shepp-Logan simulation](demo_c_3D_rigid_simulated_shepp_interleaved.ipynb)
- [3D non-rigid Shepp-Logan simulation](demo_d_3D_nonrigid_simulated_shepp_random.ipynb)

Run demos from the repository root because their data and configuration paths are relative. Select `/opt/conda/envs/environment_grics/bin/python` in Docker, or the Python interpreter from your native pip installation environment, as the notebook kernel. Set `runtime_device = "cpu"` or `"gpu"` in `config/general.toml` and check notebook overrides.

Simulations and some reconstruction steps use random initialization; `seed_enabled` and `seed` in `config/general.toml` control reproducibility. Results can vary with the seed and compute backend. Generated results are written under `runs/` by default; see the [output guide](docs/usage/outputs.md).

Run the validation scripts before using the software for a new environment or workflow. Run all four with the standard configuration and 20 simulations per test to check the installed libraries across the complete simulated reconstruction path:

```bash
python tests/validate_shepp_logan_2d_nonrigid.py --runs 20 --device gpu
python tests/validate_shepp_logan_2d_rigid.py --runs 20 --device gpu
python tests/validate_shepp_logan_3d_nonrigid.py --runs 20 --device gpu
python tests/validate_shepp_logan_3d_rigid.py --runs 20 --device gpu
```

Use `--device cpu` in every command for a CPU installation. Keep the standard reconstruction iteration schedules; `--runs 20` counts simulations, not solver iterations. Confirm that each test completes and reports `Overall quality: PASSED`; a zero exit status alone does not confirm the quality checks passed. See the [test and simulation-validation guide](docs/usage/tests.md) for output locations, device fallback notices, and result details.

For Siemens users, run `python tests/validate_siemens_phantom_2d.py --device gpu` to check Siemens conversion, reconstruction, and DICOM export; use `--device cpu` for CPU execution. See the [Siemens phantom test](docs/usage/tests.md#siemens-phantom-check).

## Performance changes (this fork)

This repository is a fork of [IADI-Nancy/GRICS-torch](https://github.com/IADI-Nancy/GRICS-torch) that makes the
GPU reconstruction 2.5 to 9 times faster on an A100 without changing the algorithm, the regularisation, the
iteration counts or the stopping rules: coils are batched into one FFT per motion state, warps and their
gradients are computed once per Gauss-Newton iteration, sparse warps are emitted pre-sorted, and the CG solver
reads its scalars once per iteration. With the default settings the reconstructed images are identical to
upstream (relative difference 0 to 3e-15). Two optional keys in the reconstruction TOML,
`compute_precision = "single"` and `warp_backend = "gather"`, trade a 1e-6 relative difference for a further
gain and 2.7x smaller stored operators. Details, measurements and figures: [docs/performance.md](docs/performance.md);
reproduction: [benchmarks/README.md](benchmarks/README.md).

## Documentation

Please check the [documentation](docs/index.md) to further usage instructions and details regarding architecture or implementation.

## Disclosure

Parts of this code and its documentation were developed with the assistance of AI tools (mostly Codex). All content has been reviewed and validated by a human.

## References

[1] Odille, F., Vuissoz, P. A., Marie, P. Y., & Felblinger, J. (2008). Generalized reconstruction by inversion of coupled systems (GRICS) applied to free‐breathing MRI. Magnetic Resonance in Medicine, 60(1), 146-157.

[2] [CuPy installation and CUDA requirements](https://docs.cupy.dev/en/v13.6.0/install.html).
