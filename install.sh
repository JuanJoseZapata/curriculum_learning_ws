#!/bin/bash

# python essentials
apt install -yq python3 python3-pip cython3 python3-ipython

python3 -m pip install --no-cache-dir --upgrade pipenv

# install CUDA 11 
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
apt update
apt install -y cuda-toolkit-12-1

# install mujoco binaries
mkdir ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.3.2/mujoco-2.3.2-linux-x86_64.tar.gz
tar -xvf mujoco-2.3.2-linux-x86_64.tar.gz

# python packages
# pip install -r requirements.txt
