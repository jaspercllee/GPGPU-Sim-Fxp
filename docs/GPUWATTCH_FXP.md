# GPUWattch fixed-point activity

This is a model repair, not a calibrated fixed-point measurement and not a result from the 2020 paper.

`FXP_INT` was missing from `perf_count_t`, so a power-enabled build failed in `set_inst_power`. The stock GTX480 XML has no `FXP_INT` entry. The coefficient array is filled by parameter name, then initialized to 1. Leaving the new slot at 1 while integer and FP instructions use `FP_INT` (4.57 in `configs/GTX480/gpuwattch_gtx480.xml`) would make one FXP instruction look cheaper than one integer instruction for no measured reason.

The rule in `src/gpuwattch/fxp_power_policy.h` is: if the XML names `FXP_INT`, use that number; otherwise copy `FP_INT`. Nothing multiplies by operand width.

Decode classification is exclusive. A fixed-point `call` is `FXP_OP` and is not also `INT_OP` or `FP_OP`. The surrounding `shr`, `add`, and compares stay ordinary integer instructions and use the integer ALU counters. `CALL_OP` does not take an integer-ALU access in `incexecstat`.

The GTX480 template is an inorder core (`machine_type` 1) at 700 MHz, matching `-gpgpu_clock_domains`. There is no FXP ALU in McPAT. Two existing structures see the count:

- The integer instruction window already adds `fxp_instructions` once, beside integer and FP counts.
- `fxpdcl` is a dependency checker, the same class as the integer and FP checkers. Its compare width is the integer architectural register index (`archi_Regs_IRF_size`, 32768, so 15 bits), because the PTX values are `.b32`. Its runtime power is now added on the inorder path. It was previously stored and ignored.

Inorder rename widths were uninitialized. They are now the architectural index widths. That defines the integer and FP checkers as well.

Power numbers require timing simulation. Functional mode does not advance the cycle clock used by GPUWattch. `scripts/run_fxp_cnn_power.sh` runs one test image with power on and checks the raw tensors against the CPU contract. It does not compare against a float GPU run and does not claim joules per image for the full test set. The recorded run is `experiments/legacy_ptx/cnn_one_image_power.json`. Kernels shorter than the 10000-cycle sample (`kprobe`, `kreq`) have an empty power block. `CONST_DYNAMIC` is constant clocking power in the GTX480 template, not a fixed-point ALU.
