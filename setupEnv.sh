#!/bin/bash

# Exit immediately if any command fails
# set -e

echo "ðŸ”§ Installing Python packages from requirements.txt..."
pip install -r requirements.txt

# echo "ðŸ” Checking if 'tinycudann' (tiny-cuda-nn) is already installed..."
# if python -c "import tinycudann" &> /dev/null; then
#     echo "âœ… 'tinycudann' is already installed. Skipping installation."
# else
#     echo "ðŸ“¦ Installing tiny-cuda-nn with PyTorch bindings..."
#     pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# fi

pip install monai

# echo "ðŸ”„ Updating Git submodules..."
# git submodule update --init --recursive


# install bart toolbox
echo "ðŸ”§ Installing BART toolbox..."
sudo apt-get update && sudo apt-get -y install make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev
cd /home/pyuser/wkdir
wget https://github.com/mrirecon/bart/archive/refs/tags/v0.9.00.tar.gz
tar xzvf v0.9.00.tar.gz
cd bart-0.9.00
make
cd /home/pyuser/wkdir
rm v0.9.00.tar.gz
echo 'export TOOLBOX_PATH=/home/pyuser/wkdir/bart-0.9.00' >> ~/.bashrc
echo 'export TOOLBOX_PATH=/home/pyuser/wkdir/bart-0.9.00' >> ~/.bashrc
echo 'export PATH=$TOOLBOX_PATH:$PATH' >> ~/.bashrc
echo 'export PYTHONPATH=$TOOLBOX_PATH/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
echo "âœ… BART toolbox installed successfully."

# Path to BART
BART_PATH="/home/pyuser/wkdir/bart-0.9.00"

# Target script file
ENV_FILE="/etc/profile.d/bart.sh"

# Check if run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root using sudo:"
  echo "sudo $0"
  exit 1
fi

# Create the environment file
echo "Creating $ENV_FILE..."

cat <<EOF > "$ENV_FILE"
export TOOLBOX_PATH=$BART_PATH
export PATH=\$TOOLBOX_PATH:\$PATH
export PYTHONPATH=\$TOOLBOX_PATH/python:\$PYTHONPATH
EOF

# Set appropriate permissions
chmod 644 "$ENV_FILE"


echo "âœ… Environment setup complete!"
echo "ðŸš€ You can now run the project."