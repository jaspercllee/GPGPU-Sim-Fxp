# GPUWattch fixed-point activity

This is a model repair, not a calibrated fixed-point measurement and not a result from the 2020 paper.

`FXP_INT` was missing from `perf_count_t`, so a power-enabled build failed in `set_inst_power`. The stock GTX480 XML has no `FXP_INT` entry. The coefficient array is filled by parameter name, then initialized to 1. Leaving the new slot at 1 while integer and FP instructions use `FP_INT` (4.57 in `configs/GTX480/gpuwattch_gtx480.xml`) would make one FXP instruction look cheaper than one integer instruction for no measured reason.

The rule in `src/gpuwattch/fxp_power_policy.h` is: if the XML names `FXP_INT`, use that number; otherwise copy `FP_INT`. Nothing multiplies by operand width. `FP_INT` is the stock scale on the instruction-window statistic. It is not joules per arithmetic operation.

Decode classification is exclusive. A fixed-point `call` is `FXP_OP` and is not also `INT_OP` or `FP_OP`. The trace keeps `FXP_INT` separate from `FP_INT`. The surrounding `shr`, `add`, and compares stay ordinary integer instructions and use the integer ALU counter in `incexecstat`. `CALL_OP` does not. The fixed-point multiply and add inside the simulator call therefore do not appear as SP, SFU, or FPU accesses. McPAT has no fixed-point functional unit, so that arithmetic energy is not in the model.

What the count does change:

- The integer instruction window adds `fxp_instructions` once, next to the integer and FP counts. That is issue-window activity.
- `fxpdcl` is a dependency checker, the same class as the integer and FP checkers. Its compare width is the integer architectural register index (`archi_Regs_IRF_size`, 32768, so 15 bits), because the PTX values are `.b32`. Its runtime power is included on the inorder path. It was previously stored and ignored. This is not an ALU.

`FXP_INT` is appended in `perf_count_t`, so the older coefficient indexes do not move. The label list ends with the same name. `scaling_coefficients` has 64 slots. The XML lookup is by parameter name. The GTX480 file has no `FXP_INT` entry.

Inorder rename widths were uninitialized. They are now the architectural index widths. That defines the integer and FP checkers as well.

## What the report numbers are

`update_components_power` does `GPUenergy += sample_power_component * executionTime`. `reset_counters` clears the per-kernel averages and does not clear `GPUenergy`. `print_power_kernel_stats` prints `GPUenergy * 1e12` at every kernel dump and labels the column pJ. Each named kernel's GPU figure is the running total through that dump. Adding them counts earlier samples again. The blank dumps that follow a kernel repeat the same total after the kernel counters were reset.

`Processor::compute` sets `executionTime` to `XML->sys.total_cycles / (clock_rate * 1e6)`. `init_mcpat` sets `total_cycles` to the sample interval, 10000. Both the XML clock and `-gpgpu_clock_domains` are 700 MHz, so one sample window is `10000/7e8` seconds. A sample is taken when `(gpu_tot_sim_cycle + gpu_sim_cycle)` is a multiple of 10000. The first call returns before any sample. Cycles after the last multiple are not in `GPUenergy` until a later sample, which can fall in the next kernel. A window can therefore contain the tail of the previous kernel. Per-kernel joules are not separated in this report.

The accumulated energy is dynamic energy plus `CONST_DYNAMIC`, the template's constant clocking term. Subthreshold leakage and gate leakage are not added. `kernel_avg_power` is the mean of the per-sample watts since the last reset, but it is printed after `setprecision(0)`, so a file value of 14 is not 14.000 W.

`scripts/run_fxp_cnn_power.sh` runs one test image in timing mode and checks the raw tensors against the CPU contract. The record is `experiments/legacy_ptx/cnn_one_image_power.json`. It keeps the last cumulative energy, not a sum of kernel rows. Functional mode is not a power result. This is not a float comparison and not a 2020-paper result.
