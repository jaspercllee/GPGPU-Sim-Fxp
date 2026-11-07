# What changed since the 2020 paper

The paper is not stored in this repository, and this change does not edit or quote it. Three kinds of statement are kept apart below. A sentence in one section is not evidence for a sentence in another.

## Three layers

1. Snapshot `3107bc1` (2020-01-08): imported simulator plus fixed-point code. Multiplication kept the left exponent and scaled by `pow(2, lhs_exp)` in float. `sub` and `div` parsed and then asserted.
2. Maintenance commit `4fe5067`: moved add/mul/sign-padding into `spirv_fxp_arith.cc`, added CPU tests and CI, and locked that old multiply result (7×8 at exponent −2 was raw 14 and exponent −2). Formulas were not changed.
3. The arithmetic update on this same branch: `binary_mul` now returns the raw product and the exponent sum (7×8 at −2,−2 is raw 56 and exponent −4). `binary_sub` and `binary_div` are implemented. Division is not described by the paper. Old accuracy and energy figures are not results of this arithmetic.

## Changes made by the maintenance commit

These packaging edits are relative to commit `3107bc1`. The multiply formula described in that commit was later replaced; see the third layer above.

- `binary_add`, `sign_padding`, and `binary_mul` moved from `spirv_device_runtime.cc` into `src/cuda-sim/spirv_fxp_arith.cc`. At that commit the arithmetic statements were the same. `binary_add` still computes `pow(2, ret_exp)` and does not apply it.
- `tests/cpu_fxp/` builds those routines and the name parser with `-O3 -DNDEBUG` and checks results with ordinary conditionals. GitHub Actions workflow `.github/workflows/cpu-fxp.yml` runs that target. It does not install CUDA.
- `scripts/run_cpu_fxp_baseline.sh` prints the on-disk version strings, lists `configs/`, and runs the CPU tests. It does not launch GPGPU-Sim.
- `README.md`, this file, `docs/FIXED_POINT.md`, and `docs/MODERNIZATION.md` describe the tree. A short pointer was added at the top of the upstream `README`. The rest of that file is unchanged.
- `tests/spirv/Makefile` gained a comment. The 2020 gtest multiply assertions expect the raw product. They disagreed with the float-scaling body until the later arithmetic update.
- `version_detection.mk` runs `nvcc --version` only when that binary is executable, so the CPU target does not print a missing-`nvcc` error. When `nvcc` exists, the detected version string is unchanged.

This branch does not migrate the simulator to current GPGPU-Sim or Accel-Sim, does not bump the CUDA toolkit, does not implement `exp10` or `log10`, does not replace fixed-point with INT8, and does not add benchmark inputs or results. `sub` and `div` are implemented in the arithmetic update. Division is an engineering policy.

## Historical facts already in the 2020 snapshot

These were true before this PR. They come from files in commit `3107bc1`, not from a paper.

- `version` contains `GPGPU-Sim Simulator Version 3.2.2`. `CHANGES` begins `Version 3.2.3+edits (development branch) versus 3.2.3`. Both strings remain. This repository does not identify a single upstream git revision, so the base is not "exactly 3.2.2" and not "exactly 3.2.3".
- The upstream manual (`README`) says the simulator was tested with CUDA 2.3, 3.1, and 4.0. `CHANGES` mentions CUDA 5.0, 5.5, 6.0, and 7.5. Checked-in products use directory names `cuda-7000` and `cuda-7050`.
- Fixed-point execution is a PTX `call` whose name contains `fxp`. `m_is_fxp` and `FXP_OP` are labels on that call. The pipeline opcode remains `CALL_OPS` with default latency 1.
- `fxp_sub` and `fxp_div` parse. `gpgpusim_spirv_invokeFxpBinaryFunction` then executes `assert(false && "Operation not supported yet")`.
- `exp10` and `log10` parse. `gpgpusim_spirv_invokeFxpMathFunction` handles `exp` and `log` only, then `assert(false && "Math function not supported yet")`.
- No device compiler, `.cu`, `.cl`, or `.ptx` workload is in the snapshot. The simulator can recognize the mangled names. It cannot show where an application would generate them.
- In snapshot `3107bc1`, `tests/spirv/fixed_point_arithmetic_impl.cc` expects a raw product and the sum of the exponents, while `binary_mul` kept the left exponent and scaled by a float power of two. The later arithmetic update follows the raw-product expectation and still narrows a 32×32 product to 32 bits.
- `gpgpu_sim_wrapper.cc` uses `FXP_INT`. `perf_count_t` in `XML_Parse.h` has no `FXP_INT` enumerator.

## Not a comparison with the paper

The paper is not committed in this repository. A private LaTeX source was read to choose the helper contracts and was not copied here. The following are still not established:

- That the stored 32-bit multiply narrowing matches the paper's width-sum. The source's multiplication branch uses an exponent sum and a width sum. This tree stores the exponent sum and, for a 4-byte return, the low 32 bits.
- Pipeline latency or the power model the paper used for its figures.
- Whether the paper's experiment executed `div`, `exp10`, or `log10`. The source's binary algorithm names addition, subtraction, and multiplication. It does not specify division.
- Any cycle count, IPC, energy, speedup, or accuracy figure. No such measurement is checked in, and none was produced here.
- A precise upstream GPGPU-Sim or Accel-Sim commit that the paper used.
- Equivalence between this fixed-point format and INT8.

If a later commit on this branch adds documentation, it should stay inside this repository and should keep new claims in the section that matches their evidence.
