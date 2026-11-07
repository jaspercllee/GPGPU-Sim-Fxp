# What changed since the 2020 paper

The paper is not stored in this repository, and this change does not edit or quote it. Three kinds of statement are kept apart below. A sentence in one section is not evidence for a sentence in another.

## Changes made by this PR

These are the only behavioral or packaging edits relative to commit `3107bc1` (`Fixed-Point data types support in GPGPU-Sim Simulator`, 2020-01-08).

- `binary_add`, `sign_padding`, and `binary_mul` moved from `spirv_device_runtime.cc` into `src/cuda-sim/spirv_fxp_arith.cc`. The arithmetic statements are the same. `binary_add` still computes `pow(2, ret_exp)` and does not apply it; the new file marks that temporary as unused with `(void)converter`.
- `tests/cpu_fxp/` builds those routines and the name parser with `-O3 -DNDEBUG` and checks results with ordinary conditionals. GitHub Actions workflow `.github/workflows/cpu-fxp.yml` runs that target. It does not install CUDA.
- `scripts/run_cpu_fxp_baseline.sh` prints the on-disk version strings, lists `configs/`, and runs the CPU tests. It does not launch GPGPU-Sim.
- `README.md`, this file, `docs/FIXED_POINT.md`, and `docs/MODERNIZATION.md` describe the tree. A short pointer was added at the top of the upstream `README`. The rest of that file is unchanged.
- `tests/spirv/Makefile` gained a comment. The 2020 gtest sources were not edited, so their multiply expectations still disagree with `binary_mul`.
- `version_detection.mk` runs `nvcc --version` only when that binary is executable, so the CPU target does not print a missing-`nvcc` error. When `nvcc` exists, the detected version string is unchanged.

This PR does not migrate the simulator to current GPGPU-Sim or Accel-Sim, does not bump the CUDA toolkit, does not implement `sub`, `div`, `exp10`, or `log10`, does not replace fixed-point with INT8, and does not add benchmark inputs or results.

## Historical facts already in the 2020 snapshot

These were true before this PR. They come from files in commit `3107bc1`, not from a paper.

- `version` contains `GPGPU-Sim Simulator Version 3.2.2`. `CHANGES` begins `Version 3.2.3+edits (development branch) versus 3.2.3`. Both strings remain. This repository does not identify a single upstream git revision, so the base is not "exactly 3.2.2" and not "exactly 3.2.3".
- The upstream manual (`README`) says the simulator was tested with CUDA 2.3, 3.1, and 4.0. `CHANGES` mentions CUDA 5.0, 5.5, 6.0, and 7.5. Checked-in products use directory names `cuda-7000` and `cuda-7050`.
- Fixed-point execution is a PTX `call` whose name contains `fxp`. `m_is_fxp` and `FXP_OP` are labels on that call. The pipeline opcode remains `CALL_OPS` with default latency 1.
- `fxp_sub` and `fxp_div` parse. `gpgpusim_spirv_invokeFxpBinaryFunction` then executes `assert(false && "Operation not supported yet")`.
- `exp10` and `log10` parse. `gpgpusim_spirv_invokeFxpMathFunction` handles `exp` and `log` only, then `assert(false && "Math function not supported yet")`.
- No device compiler, `.cu`, `.cl`, or `.ptx` workload is in the snapshot. The simulator can recognize the mangled names. It cannot show where an application would generate them.
- `tests/spirv/fixed_point_arithmetic_impl.cc` expects multiply to keep a raw product and the sum of the exponents. `binary_mul` keeps the left exponent and scales by a float power of two. Both files arrived in the same 2020 commit.
- `gpgpu_sim_wrapper.cc` uses `FXP_INT`. `perf_count_t` in `XML_Parse.h` has no `FXP_INT` enumerator.

## Not a comparison with the paper

The paper text was not available in this repository while this note was written. The following are therefore not established, and this PR does not assert them:

- Which multiply rule, pipeline latency, or power model the paper describes.
- Whether the paper states that subtract, divide, `exp10`, or `log10` are implemented.
- Any cycle count, IPC, energy, speedup, or accuracy figure. No such measurement is checked in, and none was produced here.
- A precise upstream GPGPU-Sim or Accel-Sim commit that the paper used.
- Equivalence between this fixed-point format and INT8.

If a later commit on this branch adds documentation, it should stay inside this repository and should keep new claims in the section that matches their evidence.
