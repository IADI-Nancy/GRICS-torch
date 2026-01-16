FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

COPY conda_environment.yml /tmp/conda_environment.yml

# -------------------------------------------------------------------
# Install system dependencies
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Install system dependencies (pinned versions)
# -------------------------------------------------------------------
RUN apt update \
    && apt install -y --no-install-recommends \
        wget=1.21.2-2ubuntu1.1 \
        unzip=6.0-26ubuntu3.2 \
        git=1:2.34.1-1ubuntu1.15 \
        openssh-client=1:8.9p1-3ubuntu0.13 \
        sudo=1.9.9-1ubuntu2.5 \
        ca-certificates=20240203~22.04.1 \
        bzip2=1.0.8-5build1 \
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
    pandas==2.3.3



ENV CUDA_PATH=/usr/local/cuda-12.1
ENV NVCC=$CUDA_PATH/bin/nvcc
ENV PATH=$CUDA_PATH/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# -------------------------------------------------------------------
# Install BART
# -------------------------------------------------------------------
# Download and extract
RUN mkdir -p /opt/bart \
    && wget https://github.com/mrirecon/bart/archive/refs/tags/v0.9.00.tar.gz \
    && tar xzvf v0.9.00.tar.gz -C /opt/bart --strip-components=1 \
    && rm v0.9.00.tar.gz \
    && cd /opt/bart \
    && CUDA=1 \
        CUDNN=1 \
        CUDA_PATH=$CUDA_PATH \
        NVCC=$NVCC \
        CC=gcc-11 \
        CXX=g++-11 \
        CPPFLAGS="-I/opt/bart/src -I$CUDA_PATH/include" \ 
        CFLAGS="-I/opt/bart/src -I$CUDA_PATH/include" \
        LDFLAGS="-L$CUDA_PATH/lib64" \
        make -j$(nproc) 

# Add BART to PATH
ENV BART_TOOLBOX_PATH=/opt/bart
ENV PATH=/opt/bart:$PATH
ENV PYTHONPATH="/opt/bart/python" 
# Force single-threading for BART (speeds up slice-wise operations)
ENV OMP_NUM_THREADS=1

# -------------------------------------------------------------------
# Install h5py wrapper for SAEC files
# -------------------------------------------------------------------
RUN git clone https://github.com/IADI-Nancy/wrapperHDF5.git /opt/wrapperHDF5
ENV PYTHONPATH=/opt/bart/python:/opt/wrapperHDF5
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Startup (needed for usage in IADI lab network)
# -------------------------------------------------------------------

RUN echo 'pyuser ALL=(ALL) NOPASSWD: ALL\n\
Defaults    env_keep += "http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY"\n\
Defaults    env_keep += "PATH"' >> /etc/sudoers \
	&& sed -i '/^Defaults\tsecure_path.*$/ d' /etc/sudoers \
	&& echo '[distutils]\n\
index-servers =\n\
    pypi\n\
    virtus\n\
\n\
[pypi]\n\
repository: https://pypi.python.org/pypi\n\
username:\n\
password:\n\
\n\
[virtus]\n\
repository: http://virtus.iadi.lan:8081/repository/virtus-pip/\n\
username:\n\
password:' >> /root/.pypirc \
	&& mkdir -p /root/.config/pip/ \
	&& echo '[global]\n\
index = https://pypi.python.org/pypi/\n\
index-url=https://pypi.python.org/simple/\n\
extra-index-url=http://virtus.iadi.lan:8081/repository/virtus-pip/simple/\n\
trusted-host = virtus.iadi.lan' >> /root/.config/pip/pip.conf

COPY startup.sh /tmp
RUN chmod 774 /tmp/startup.sh

ENTRYPOINT ["/tmp/startup.sh"]
