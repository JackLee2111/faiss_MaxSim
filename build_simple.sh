#!/bin/bash
# Build FAISS from source inside a conda environment

set -e  # fail fast

cd /data1/lijunlin/faiss
rm -rf build

# Activate conda env and set paths
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/lijunlin/faiss/build/faiss:$LD_LIBRARY_PATH


# Install necessary tools (optional, if not already installed)
conda install -y gxx_linux-64 gcc_linux-64 cmake swig numpy

# Reconfigure and rebuild

cmake -B build \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_GPU=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DPython_EXECUTABLE=$(which python) \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX


cmake --build build -j$(nproc)
# cmake --install build

# root of faiss
pip install ./build/faiss/python/

# Install Python bindings into conda env, NOT user
cd build/faiss/python
python setup.py install

# Test installation
# python -c "import faiss; print(faiss.__file__)"

# ----------- end of build_simple.sh -----------





# # Build script for FAISS with Python bindings
# cmake -DFAISS_ENABLE_PYTHON=ON \
#       -DFAISS_ENABLE_GPU=OFF \
#       -DCMAKE_BUILD_TYPE=Release \
#       -DPYTHON_EXECUTABLE=$(which python3) ..

# # cmake .. -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release

# make -j$(nproc)
# make install

# # 这一步会将 faiss 安装到当前 Python 环境中，且链接的是你刚刚编译的 C++ 动态库（含源码改动）。
# cd faiss/python && python setup.py install

