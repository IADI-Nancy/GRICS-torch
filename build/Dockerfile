FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL org.opencontainers.image.source=https://github.com/IADI-Nancy/GRICS-torch
LABEL org.opencontainers.image.description="GRICS-torch: A PyTorch-based framework for MRI reconstruction with motion compensation, developed by the IADI lab in Nancy, France."
LABEL org.opencontainers.image.licenses="CC BY-NC 4.0"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

COPY conda_environment.yml /tmp/conda_environment.yml

# -------------------------------------------------------------------
# Install system dependencies
# -------------------------------------------------------------------
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list \
    && apt update \
    && apt install -y --no-install-recommends \
        wget \
        unzip \
        git \
        openssh-client \
        sudo \
        ca-certificates \
        bzip2 \
        build-essential=12.9ubuntu3 \
        cmake=3.22.1-1ubuntu1.22.04.2 \
        liblapacke-dev=3.10.0-2ubuntu1 \
        libopenblas-dev=0.3.20+ds-1 \
    && apt install -y --no-install-recommends \
        libfftw3-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Install Miniconda and conda environment
# -------------------------------------------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

 # Accept Anaconda ToS for default channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda config --remove channels defaults \
 && conda config --add channels conda-forge \
 && conda config --set channel_priority strict \
 && conda env create -f /tmp/conda_environment.yml \
 && conda clean -afy \
 && rm /tmp/conda_environment.yml
ENV PATH=/opt/conda/envs/environment_grics/bin:$PATH

# -------------------------------------------------------------------
# Install pip packages
# -------------------------------------------------------------------
# PyTorch (CUDA wheels)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Everything else
RUN pip install --no-cache-dir \
    ismrmrd==1.14.2 \
    GitPython==3.1.46 \
    pandas==2.3.3 \
    scikit-image==0.26.0

ENV CUDA_PATH=/usr/local/cuda-12.1
ENV NVCC=$CUDA_PATH/bin/nvcc
ENV PATH=$CUDA_PATH/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# -------------------------------------------------------------------
# Install h5py wrapper for SAEC files
# -------------------------------------------------------------------
RUN git clone https://github.com/IADI-Nancy/wrapperHDF5.git /opt/wrapperHDF5
ENV PYTHONPATH=/opt/wrapperHDF5
# -------------------------------------------------------------------

RUN echo 'pyuser ALL=(ALL) NOPASSWD: ALL\n\
Defaults    env_keep += "PATH"' >> /etc/sudoers \
	&& sed -i '/^Defaults\tsecure_path.*$/ d' /etc/sudoers

COPY startup.sh /tmp
RUN chmod 774 /tmp/startup.sh

ENTRYPOINT ["/tmp/startup.sh"]
