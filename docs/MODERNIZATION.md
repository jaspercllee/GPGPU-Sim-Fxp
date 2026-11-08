# Modernization roadmap

This is a plan for later work. Phase 0 is the only phase implemented in the repository. Later phases are not started, and this document does not report measured speedups, paper figures, or an upstream commit that is not in the tree.

## Phase 0 — baseline that can run without CUDA (this change)

- Document the `version` (3.2.2) versus `CHANGES` (3.2.3+edits) discrepancy.
- Move `binary_add`, `binary_mul`, and `sign_padding` into `spirv_fxp_arith.cc` so a host test can call them. That first maintenance commit did not rewrite formulas. A later commit on this branch changes only multiplication, and adds subtraction and division. Addition is still the 2020 routine, including the unused `pow` result.
- Add `tests/cpu_fxp`, built with `-O3 -DNDEBUG`, and GitHub Actions workflow `.github/workflows/cpu-fxp.yml`.
- Add `scripts/run_cpu_fxp_baseline.sh`. It prints the version strings, lists `configs/`, and runs the CPU tests. It does not launch the simulator.

Not done in this phase: rebuilding GPGPU-Sim, running CUDA or OpenCL, Accel-Sim, a newer CUDA toolkit, or any benchmark.

## Phase 1 — make the existing 3.x tree build again

Do this before any architecture migration. Keep the same fixed-point numbers.

- Confirm a host compiler and one CUDA toolkit that can still configure this tree. The upstream README cites CUDA 2.3/3.1/4.0. `CHANGES` and the `cuda-7000` / `cuda-7050` directories point at 7.0 and 7.5. Pick one and record the exact toolkit path. Do not claim a newer toolkit works until `libcudart.so` links and a real kernel is run.
- Decide what to do with `FXP_INT`. Using it does not compile against `perf_count_t`. Adding an enumerator changes the power-coefficient table layout, so it needs a separate check against `gpuwattch` XML, not a silent insert.
- `sub` and `div` are implemented in `spirv_fxp_arith.cc`. `exp10` and `log10` still parse and then hit `assert` in `gpgpusim_spirv_invokeFxpMathFunction`.
- Remove or gate the debug `printf` calls in `call_impl` (`1519: call_impl_in_binary` and the convert/math lines) only after confirming no workflow depends on that text.
- Multiplication on this branch now uses the raw product and the exponent sum. A 4-byte return still keeps the low 32 bits. The paper's width-sum storage for 32×32 is 64 bits and is not what the helper stores.
- Stop committing `build/` and `lib/` objects once a clean build is reproducible. That should be its own commit so the functional diff stays reviewable.

## Phase 2 — reproducible simulator runs

Blocked until someone supplies the workloads. This tree has configuration files and no `.ptx`, `.cu`, or `.cl` applications.

- Record the application source, the CUDA toolkit used to compile it, the `configs/<gpu>/` directory, `PTX_SIM_MODE_FUNC`, and the expected functional checksum.
- A timing run is a second script. It must print the simulator version string from `version` and must not invent a cycle count.
- OpenCL needs the NVIDIA OpenCL headers and library that `setup_environment` looks up. Those are not in the repository.

## Phase 3 — newer CUDA on this 3.x tree

Only after Phase 1. GPGPU-Sim 3.x's `libcuda/cuda_runtime_api.cc` is a hand-written subset of the runtime. Newer driver APIs (`cuLaunchKernel`, fatbin compression, newer PTX) will not start working because the host compiler is newer. Track missing entry points from a real application instead of bumping a version macro.

## Phase 4 — current GPGPU-Sim or Accel-Sim

Separate project. Accel-Sim and GPGPU-Sim 4.x replace the SASS/PTX tracing and configuration model. Porting means re-implementing the fixed-point call hooks (`m_is_fxp`, `FXP_OP`, `call_impl`, `spirv_fxp_arith`) on that code base and re-checking every CPU baseline case. Keep this repository as the frozen 3.x baseline until that port has its own tests. Do not describe Phase 4 as done from this tree.

## Fixed-point and INT8

The implemented format is a signed magnitude stored in 8, 16, or 32 bits, plus an exponent carried in the mangled name (`fxp_<width>_<abs-exp>_<metadata>`). Metadata bit 0 marks a signed accumulator and bit 1 marks a negative exponent (`FXP_META_MASK_ACU_SIGNED`, `FXP_META_MASK_EXP_NEGATIVE`).

INT8, as used by current tensor cores or integer GEMM, is a different contract: scale, zero-point, and saturation live outside these helpers. Replacing these helpers with INT8 would change results. If an INT8 mode is added later, put it behind a new function name and keep `binary_add` / `binary_mul` as they are.

`binary_add` aligns to the smaller exponent by shifting, then adds in the wider of the two storage sizes. The 8- and 16-bit paths add as unsigned; the 32-bit path adds as `int32_t`. `binary_mul` sign-extends, multiplies in `int64`, and stores the low bits of the historical return width. `binary_sub` and `binary_div` are specified in [FIXED_POINT.md](FIXED_POINT.md). There is no saturation flag on the arithmetic path (`FxpConvertFunction::Saturation` exists and is left false).

## Repository facts

Code facts and the split between this PR, the 2020 snapshot, and unverified paper comparisons are in [SINCE_2020.md](SINCE_2020.md) and [FIXED_POINT.md](FIXED_POINT.md). The list below is only a pointer. It is not a measurement and it is not a claim about the paper text.

These are facts from the source tree. They are not measurements.

- The functional hook is a PTX `call` whose callee name contains `fxp`. `ptx_instruction` sets `m_is_fxp`, and `set_fp_or_int_archop` labels that call `FXP_OP`.
- Timing still classifies the instruction as `CALL_OPS`. The opcode setup leaves the default latency and initiation interval at 1. `sp_unit` accepts `CALL_OPS`. There is no separate fixed-point pipeline stage in `shader.h`.
- Decode counts `m_num_FXPdecoded_insn` when `oprnd_type == FXP_OP`. `power_stat_t::get_total_fxp_inst` forwards that count. `gpgpu_sim_wrapper::set_inst_power` writes `fxp_instructions`. The `FXP_INT` coefficient index is missing, so this power path is incomplete in source.
- GPUWattch XML and `core.cc` mention `phy_Regs_FXPRF_size`, `archi_Regs_FXPRF_size`, and `fxp_instructions`. That is a power-model counter and register-file field, not evidence that the shader core executes fixed-point arithmetic on a dedicated unit.
- Supported call implementations: add, mul, convert (`rte`, `rtn`, `rtz`, `rtp`), `exp`, and `log`. `exp` and `log` convert through `float` (`std::exp`, `std::log`). `sub`, `div`, `exp10`, and `log10` are recognized and not implemented.
- The name `spirv` is the C++ namespace. The accepted strings are `fxp_add` / `convert_` / `exp` mangled names. The gtest file rejects `_Z14__spirv_FxpAdd...`.
- No application, PTX dump, or result log in this repository supplies a cycle count, IPC, or energy number.
- Git authorship on the fixed-point commit is `clli <clli@pllab.cs.nthu.edu.tw>`, 2020-01-08, message `Fixed-Point data types support in GPGPU-Sim Simulator`. That identifies the snapshot; it does not identify an upstream GPGPU-Sim hash.

## Decisions that need the repository owner

- Whether `exp10` and `log10` should be implemented. `sub` and `div` now have CPU contracts; division remains an engineering policy because the paper does not specify it.
- Whether a fresh-weight MNIST MLP reconstruction should be started. That proposal is in [RECONSTRUCTION_PLAN.md](RECONSTRUCTION_PLAN.md). It is not a claim that the paper used a CNN, and it is not permission to publish new accuracy numbers.
- Which CUDA toolkit is allowed for a Phase 1 rebuild.
- Whether committed `build/` and `lib/` objects should be removed.
- Whether the next architecture port is GPGPU-Sim 4.x, Accel-Sim, or staying on this 3.x tree.
- Where the original benchmark sources live, if a paper figure is to be regenerated.
