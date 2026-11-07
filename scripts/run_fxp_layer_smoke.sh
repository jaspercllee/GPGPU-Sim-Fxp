#!/bin/sh
# Functional, power-disabled smoke inside the Ubuntu 14.04 CUDA 7.5 chroot.
# The PTX file is a hand-written producer, not the 2020 LLVM/SPIR-V compiler.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
CHROOT=/var/tmp/ubuntu-14.04-cuda75
if [ ! -x "$CHROOT/usr/local/cuda-7.5/bin/nvcc" ]; then
  echo "Ubuntu 14.04 CUDA 7.5 chroot is not present at $CHROOT" >&2
  exit 1
fi
sudo cp "$ROOT/src/cuda-sim/spirv_device_runtime.cc" /var/tmp/gpgpu-src/src/cuda-sim/
sudo cp "$ROOT/experiments/legacy_ptx/fxp_layer_smoke.ptx" "$CHROOT/tmp/run/fxp_layer_smoke.ptx"
sudo cp "$ROOT/experiments/legacy_ptx/fxp_layer_host.cu" "$CHROOT/tmp/run/fxp_layer_host.cu"
sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /src; source setup_environment; unset GPGPUSIM_POWER_MODEL; make -j2'
sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /tmp/run; nvcc -cudart shared -arch=sm_20 fxp_layer_host.cu -o fxp_layer_host && PTX_SIM_MODE_FUNC=1 PTX_SIM_USE_PTX_FILE=1 PTX_SIM_KERNELFILE=/tmp/run/fxp_layer_smoke.ptx LD_LIBRARY_PATH=/src/lib/gcc-4.8.4/cuda-7050/release ./fxp_layer_host'
