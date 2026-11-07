# Fixed-point flow, capabilities, and limits

This note describes the code in this repository. It does not claim that a newer CUDA toolkit, current GPGPU-Sim, or Accel-Sim can run these hooks. Those ports are not implemented here.

The 2020 paper is not in this tree. Comparisons with that paper are in [SINCE_2020.md](SINCE_2020.md). Version strings are recorded in [README.md](../README.md) and are not collapsed into one upstream revision.

## Call path

There is no fixed-point PTX opcode. A kernel is expected to contain an ordinary PTX `call` to an Itanium-mangled helper. This repository does not contain the device compiler, CUDA source, OpenCL source, or PTX that would emit those calls. The accepted names are parsed from strings such as `_Z7fxp_add12fxp_8_3_3_12fxp_16_3_3_`. A SPIR-V-looking name such as `_Z14__spirv_FxpAdd...` is rejected by `IsFxpBinaryFunction`. The C++ namespace is `spirv` anyway.

1. The CUDA or OpenCL front end (`libcuda/cuda_runtime_api.cc`, `libopencl/opencl_runtime_api.cc`) launches a grid. `PTX_SIM_MODE_FUNC` selects the pure functional loop. Otherwise `gpgpu_sim::cycle` runs the timing model.
2. PTX text is loaded by `gpgpu_ptx_sim_load_ptx_from_string` and parsed by the existing PTX parser. No fixed-point token was added to `ptx.l` / `ptx.y`.
3. When the callee name contains the substring `fxp`, `ptx_instruction` sets `m_is_fxp`. `set_fp_or_int_archop` then labels that `CALL_OP` as `FXP_OP`.
4. The timing class stays `CALL_OPS`. Opcode setup leaves latency and initiation interval at the default of 1. `sp_unit` can issue `CALL_OPS`. `shader.h` has no fixed-point functional unit.
5. Both the functional simulator and the timing issue path reach `ptx_thread_info::ptx_exec_inst`, which dispatches `call_impl`.
6. `call_impl` classifies the callee:
   - `IsFxpBinaryFunction` → `gpgpusim_spirv_invokeFxpBinaryFunction`
   - `IsFxpConvertFunction` → `gpgpusim_spirv_invokeFxpConvertFunction`
   - `IsFxpMathFunction` → `gpgpusim_spirv_invokeFxpMathFunction`
7. Add and multiply are implemented in `src/cuda-sim/spirv_fxp_arith.cc`. Convert, `exp`, and `log` remain in `spirv_device_runtime.cc` and still need the simulator types (`ptx_thread_info`, local memory). The CPU tests call only the parser and `spirv_fxp_arith`.

Decode counts `m_num_FXPdecoded_insn` for `FXP_OP`. `get_total_fxp_inst` forwards that count into GPUWattch's `fxp_instructions` field. `set_inst_power` multiplies by `scaling_coefficients[FXP_INT]`, and `FXP_INT` is not a member of `perf_count_t`. That power path is incomplete in the current sources. GPUWattch also has `FXPRF` size fields. Those fields are not a separate shader pipeline.

## Format

The helper name carries `fxp_<width>_<absolute exponent>_<metadata>`.

- Storage width in the arithmetic helpers is 8, 16, or 32 bits. Other widths hit `assert`.
- Metadata bit 0 (`FXP_META_MASK_ACU_SIGNED` / `FXP_MASK_ACCU_SIGNED`) marks a signed accumulator.
- Metadata bit 1 (`FXP_META_MASK_EXP_NEGATIVE` / `FXP_MASK_EXP_SIGNED`) makes the exponent negative.

This is not an INT8 tensor contract. There is no zero-point, and the arithmetic path does not apply the `Saturation` flag on `FxpConvertFunction` (it is left false).

`binary_add` aligns both operands to the smaller exponent by shifting, promotes to the wider storage size, and adds. The 8-bit and 16-bit paths add as unsigned values. The 32-bit path adds as `int32_t`. An unused `pow` result is still computed and then discarded; the add itself does not scale by that float.

`binary_mul` keeps the left-hand exponent, sign-extends, multiplies in a signed 8-, 16-, or 32-bit integer, then multiplies by `pow(2, lhs_exp)` in `float` and truncates back to the integer width. Overflow wraps. The 2020 gtest file `tests/spirv/fixed_point_arithmetic_impl.cc` expects a different multiply rule (raw product and `lhs_exp + rhs_exp`). For inputs 7 and 8 at exponent -2, the gtest expects exponent -4 and raw 56. The function returns exponent -2 and raw 14. This PR does not change either side. The CPU test locks the function's result.

## What actually runs

| Call | Recognized | Executed |
| --- | --- | --- |
| `fxp_add` | yes | `binary_add` |
| `fxp_mul` | yes | `binary_mul`, with the float rescale above |
| `fxp_sub` | yes | no. `gpgpusim_spirv_invokeFxpBinaryFunction` hits `assert(false && "Operation not supported yet")` |
| `fxp_div` | yes | same assert as `fxp_sub` |
| `convert_*` with `_rte`, `_rtn`, `_rtz`, `_rtp` | yes | convert routines in `spirv_device_runtime.cc` |
| `exp`, `log` | yes | converted through `float` via `std::exp` / `std::log` |
| `exp10`, `log10` | yes | no. `gpgpusim_spirv_invokeFxpMathFunction` hits `assert(false && "Math function not supported yet")` |

`assert` is compiled out of a release simulator build (`NDEBUG`). The CPU baseline tests do not use `assert` for their checks, so those checks still run under `-O3 -DNDEBUG`. The unimplemented `sub` and `div` paths are not given a release-mode fallback here; calling them in a release simulator build would fall through the missing `assert`. That behavior was already in the 2020 sources.

## Build and use

CPU baseline, no CUDA and no GPU:

```sh
make -C tests/cpu_fxp
./scripts/run_cpu_fxp_baseline.sh
make cpu-fxp-tests
```

Requirements: `g++` (C++11) and `make`. Boost 1.63 headers ship in `third_party/boost`.

Full simulator, not run by CI and not rebuilt in the change that added these docs:

```sh
export CUDA_INSTALL_PATH=/path/to/cuda
source setup_environment
make
```

Upstream `README` lists `build-essential`, `xutils-dev`, `bison`, `zlib1g-dev`, `flex`, `libglu1-mesa-dev`, and `makedepend`. OpenCL also needs `NVOPENCL_LIBDIR` and `NVOPENCL_INCDIR`. That `README` still says the simulator was tested with CUDA 2.3, 3.1, and 4.0. `CHANGES` discusses CUDA 5.0 through 7.5, and checked-in library directories are named `cuda-7000` and `cuda-7050`. None of those statements has been re-validated on a current toolkit in this change.

`tests/spirv/` is the 2020 Google Test driver. It needs gtest 1.8, downloaded by `test.mk`, and a built `libOpenCL.so`. CI does not run it.

No `.ptx`, `.cu`, or `.cl` workload is in the tree. `configs/` holds GTX 480, Tesla C2050, Quadro FX5600, Quadro FX5800, and GeForce GTX 750 Ti configuration files only. They do not reproduce a benchmark number.
