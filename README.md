# GPGPU-Sim-Fxp

Fixed-point support on top of a GPGPU-Sim 3.x tree. The simulator still models CUDA and OpenCL at PTX level. The fixed-point work recognizes Itanium-mangled calls such as `_Z7fxp_add...` and executes them in the functional model. It does not add a new PTX opcode, and it is not an INT8 quantization mode.

This repository preserves that 2020 research baseline. Migration to current GPGPU-Sim or Accel-Sim is not done.

- Flow, operation matrix, and numeric limits: [docs/FIXED_POINT.md](docs/FIXED_POINT.md)
- Snapshot versus this branch: [docs/SINCE_2020.md](docs/SINCE_2020.md)
- Research-review notes, not new measurements: [docs/research-context.md](docs/research-context.md), [docs/experiment-rerun-plan.md](docs/experiment-rerun-plan.md)
- Fresh-weight MNIST MLP, CPU only: [docs/RECONSTRUCTION_PLAN.md](docs/RECONSTRUCTION_PLAN.md) and `scripts/run_mnist_mlp.sh`
- Later build and porting work: [docs/MODERNIZATION.md](docs/MODERNIZATION.md)

`fxp_add` keeps the 2020 routine. `fxp_mul` now returns the raw integer product and the sum of the exponents. `fxp_sub` aligns to the smaller exponent and subtracts. `fxp_div` is an extra policy: the paper does not specify division. `exp10` and `log10` are still recognized and not implemented. mad, neg, abs, min, max, compare, and shift are not in the parser. The device compiler is not in the tree.

## Version evidence in this tree

These strings disagree. Both are present; neither has been rewritten to match the other.

| Source | What it says |
| --- | --- |
| [`version`](version) | `GPGPU-Sim Simulator Version 3.2.2` |
| [`CHANGES`](CHANGES) | `Version 3.2.3+edits (development branch) versus 3.2.3`, including CUDA 5.0–7.5 notes and a GeForce GTX 750 Ti config |
| Prebuilt library directories | `lib/gcc-4.8.4/cuda-7000/` and `cuda-7050/` |
| Upstream `README` | Build text written for CUDA 2.3, 3.1, and 4.0 |

There is no upstream commit id in this git history. The first commit is only this README title. The second commit, `3107bc1` (2020-01-08), imports the simulator and the fixed-point code together. Do not cite a more precise upstream revision than the files above support, and do not flatten the two version strings into one exact base. The tree reports 3.2.2 in `version` and 3.2.3+edits in `CHANGES` at the same time.

The long upstream instructions remain in [`README`](README). Where that file and this one disagree, this file describes the tree as it exists now.

## What builds without a GPU or CUDA

```sh
make -C tests/cpu_fxp
# or
./scripts/run_cpu_fxp_baseline.sh
```

Requirements: `g++` with C++11 and `make`. Boost 1.63 headers are already in `third_party/boost` (Spirit and Phoenix). The test binary is compiled with `-O3 -DNDEBUG`. Its checks are normal conditionals, so they still run when `assert` is compiled out.

`make cpu-fxp-tests` from the repository root runs the same target. It does not need `source setup_environment` or `nvcc`. `version_detection.mk` probes `nvcc` only when `$(CUDA_INSTALL_PATH)/bin/nvcc` is executable.

The historical Google Test files under `tests/spirv/` still expect a downloaded gtest 1.8 tree and a built `libOpenCL.so`. They are not the continuous-integration path. Their multiply assertions now match the corrected raw-product contract. CI runs `tests/cpu_fxp`.

## Full simulator build

Still the upstream flow, and it is not covered by CI:

```sh
export CUDA_INSTALL_PATH=/path/to/cuda
source setup_environment
make
```

Host packages named by the upstream README: `build-essential`, `xutils-dev`, `bison`, `zlib1g-dev`, `flex`, `libglu1-mesa-dev`, plus `makedepend`. OpenCL also needs `NVOPENCL_LIBDIR` and `NVOPENCL_INCDIR`. GPUWattch is optional via `GPGPUSIM_POWER_MODEL`.

This environment was not used to rebuild `libcudart.so`. Do not treat the checked-in objects under `build/` and `lib/` as a fresh build on a current compiler.

A known source inconsistency: `gpgpu_sim_wrapper.cc` indexes `scaling_coefficients[FXP_INT]`, but `FXP_INT` is not a member of `perf_count_t` in `XML_Parse.h`. A power-model build of the current sources is not known to succeed.

## Layout

- `libcuda/`, `libopencl/`: CUDA runtime and OpenCL API stand-ins
- `src/cuda-sim/`: PTX front end, functional execution, fixed-point calls
- `src/gpgpu-sim/`: timing model (shader pipeline, caches, DRAM)
- `src/intersim2/`: BookSim 2 interconnect
- `src/gpuwattch/`: McPAT-based power model
- `cuobjdump_to_ptxplus/`: SASS to PTXPlus
- `configs/`: GTX 480, Tesla C2050, Quadro FX5600, Quadro FX5800, GeForce GTX 750 Ti
- `aerialvision/`: statistics GUI
- `tests/cpu_fxp/`: CUDA-free fixed-point baseline
- `tests/spirv/`: 2020 gtest sources; not run by CI
- `docs/FIXED_POINT.md`, `docs/SINCE_2020.md`, `docs/MODERNIZATION.md`: flow, 2020-versus-this-PR notes, and later work

No `.ptx`, `.cu`, or `.cl` workloads are checked in. Config files alone do not reproduce a benchmark result.
