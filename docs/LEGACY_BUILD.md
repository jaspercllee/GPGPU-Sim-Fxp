# Legacy CUDA 7.5 build check

This is a cloud-VM chroot log, not a container image that is committed here. It does not install a driver, does not change a user machine, and does not rerun the MNIST accuracy or energy figures.

## What was verified

- Ubuntu 16.04 (xenial) debootstrap from `http://archive.ubuntu.com/ubuntu`, gcc `(Ubuntu 4.8.5-4ubuntu2) 4.8.5`. The simulator linked, then aborted at startup with `std::ios_base::failure[abi:cxx11]` from `option_parser.cc` `fromString`. That is a libstdc++ dual-ABI mismatch with Ubuntu 16.04's newer `libstdc++.so.6`, not a successful run.
- Ubuntu 14.04 (trusty) debootstrap from the same archive, gcc `(Ubuntu 4.8.4-2ubuntu1~14.04.4) 4.8.4`, `libstdc++6 4.8.2-19ubuntu1`. This is the run that executed kernels.
- CUDA toolkit only: official runfile `https://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run`, sha256 `08411d536741075131a1858a68615b8b73c51988e616e83b835e4632eea75eec`, installed with `sh cuda_7.5.18_linux.run --silent --toolkit --override`. The installer summary was `Driver: Not Selected`, `Toolkit: Installed in /usr/local/cuda-7.5`, `Samples: Not Selected`. `nvcc` reports release 7.5, `V7.5.17`. `nvidia-smi` is not present.
- `source setup_environment` then `unset GPGPUSIM_POWER_MODEL` and `make -j2`. The default power-model build stops at `gpgpu_sim_wrapper.cc:270` because `FXP_INT` is not in `perf_count_t`. The verified `libcudart.so` is the no-GPUWattch build. Its sha256 is `c4049af0d324083bae2ffda49c9b875f64beecf2960ef32aebb8e5234eb69182`. It is not committed.
- gcc 4.8 also rejects an unqualified `isnan` in this tree. The build uses `x != x` in those six checks. That is the IEEE NaN test, not a fixed-point change.
- A 4-thread `sm_20` kernel compiled by this `nvcc` printed `OUT 1 2 3 4` under `LD_LIBRARY_PATH` pointing at the new `libcudart.so`, with `gpgpusim.config` from `configs/GTX480` and power simulation turned off in that run directory.
- One fixed-point call: nvcc 7.5 emits PTX `.version 4.3` and `.target sm_20`. The launched text is the hand-written file `experiments/legacy_ptx/fxp_mul_call.ptx`, loaded with `PTX_SIM_USE_PTX_FILE=1` and `PTX_SIM_KERNELFILE`. It is a new minimal PTX producer, not the 2020 LLVM/SPIR-V compiler. The callee is `_Z7fxp_mul12fxp_32_2_3_12fxp_32_2_3_` (32-bit raw values, exponent -2, metadata 3). The log contains `1519: call_impl_in_binary:` and `copy no error value 56`, which matches the CPU raw product `7 * 8`.

## Not verified

GPUWattch, OpenCL, a restored Clang/LLVM-SPIR-V/llc producer, CNN, and any simulator energy or accuracy number. The MNIST 8983/10000 figure remains a CPU result.
