# Experiment rerun and validation plan

**Status: proposed and unrun.** This is an evidence plan, not a benchmark result or migration implementation. Read [research context](research-context.md) for the inspected revision, historical paper results, and verified related work. Operational commands and supported toolchains belong in the maintained [README](../README.md) and maintenance test documentation.

## 1. Keep three evidence tracks separate

1. **Historical reproduction:** preserve the original snapshot/configuration/compiler/weights if recoverable. Reproduce the 2020 figures without silently changing arithmetic semantics. If artifacts are missing, call this a reconstruction, not an exact reproduction.
2. **Corrected numerical implementation:** compare an independently specified reference with both archival and corrected behavior. Any correction affecting outputs invalidates direct reuse of old accuracy/energy results; retain an explicit archival-versus-corrected comparison.
3. **Modernized execution/modeling:** pin a newer CUDA/toolchain and simulator baseline only after compatibility is demonstrated. Hardware validation applies to instructions actually executed on that hardware. A hypothetical custom unit remains a modeled design, even if the rest of the baseline is validated.

Never combine these tracks into one unlabeled “updated result.”

**Branch update after the review:** multiplication on this branch now uses a raw product and an exponent sum, with 32-bit narrowing when the return is 4 bytes. Subtraction and an explicitly non-paper division policy are implemented and covered by `tests/cpu_fxp`. That does not rerun Figures 4–6. A fresh-weight MNIST MLP, not a CNN and not a historical reproduction, is proposed in [RECONSTRUCTION_PLAN.md](RECONSTRUCTION_PLAN.md) and has not been trained.


## 2. Immediate rerun ledger

| Existing claim or artifact | Required action | Evidence needed before reuse |
|---|---|---|
| Fig. 4: FXP8 vs FXP32 DRAM energy | Reproduce and audit causality. Compare the displayed float-buffer kernel with an explicitly packed-storage variant. | Host allocation types, generated PTX loads/stores, bytes/sectors/transactions, cache misses, cycle counts, and DRAM energy accounting. |
| Fig. 5: 8%/14% total GPU energy reduction | Rerun whenever semantics, compiler, simulator, configuration, or workload changes. | Absolute energy and duration, per-component breakdown, operation counters, coefficients/units, static-power treatment, normalization baseline. Keep original figures historical until then. |
| Fig. 6: 90%/82%/74% accuracy | Evaluate identical checkpoints and identical full held-out examples under all formats. | Resolve 50-image versus 10,000-image ambiguity; publish labels, selected indices if sampled, predictions, and accuracy/confidence intervals. |
| Section 3 / Algorithm 2 semantics | Establish executable reference before experiments. | Addition alignment, raw-product/exponent-addition expectations, multiplication result width, narrowing, signedness, rounding, saturation/wrap policy, and accumulation semantics. Reconcile archived runtime with Algorithm 2 and archived gtests; no newer implementation or passing logs were inspected. |
| Performance improvement wording | Supply a new timing evaluation or remove the claim. | Kernel and end-to-end latency/throughput, instruction counts, stall breakdown, occupancy/register pressure, and conversion cost. Energy alone is not speedup. |
| New CUDA/Accel-Sim/FP8/Tensor Core claims | Require new implementation and validation, not only CI/docs/plans. | Exact versions, capability manifest, small end-to-end example, numerical checks, relevant hardware counters and correlation. |

**Additional source cross-check:** LaTeX Algorithm 2 confirms exponent/width addition for multiplication. Listing 4 uses input exponents -2/-3 and accumulator exponent -6, whereas Figure 6 labels FXP8 as `(8,-4)`. Recover the actual figure-producing kernel/configuration instead of treating the listing as that configuration. Figure 4's 85% and Figure 5's 100%/92%/86% are rounded chart labels, not exact ratios; source-PDF absolute values are recorded in [research context](research-context.md).

**Can be edited without a rerun:** terminology, historical labeling, related-work update, explanation of source-visible dispatch, limitation statements, artifact inventory, and removal/qualification of unsupported generalizations. These changes must not imply new measurements.

## 3. Functional correctness first

### Define the numerical contract

For a signed W-bit representation, document `x = q * 2^E`, `q ∈ [-2^(W-1), 2^(W-1)-1]`; specify unsigned behavior separately. Define:

- rescaling of unequal exponents before addition/subtraction;
- full product width/exponent and when the result is narrowed;
- accumulator width and whether rounding occurs after every product/add or only at an output boundary;
- overflow behavior, saturation versus wrap, and out-of-range conversions;
- RTE tie handling and negative RTN/RTP/RTZ cases;
- treatment of NaN, infinity, signed zero, and unsupported operations.

**Proposed multiplication contract, not a verified implementation:** for `x = q_lhs * 2^E_lhs` and `y = q_rhs * 2^E_rhs`, first form the raw product `q_product = q_lhs * q_rhs` and `E_product = E_lhs + E_rhs`; apply destination rescaling/narrowing separately. This agrees with the paper's multiplication branch on raw product/exponent and with the archived `BinaryMulBasic` gtest (7 × 8 → raw 56, exponent -4). However, Algorithm 2 sums input widths while archived tests request a 32-bit return and one explicitly expects wrapped overflow. Resolve that difference rather than claiming full paper/test equivalence. The pinned runtime instead uses a left-operand exponent and float-based scaling. A relayed design-direction report does not establish owner approval, changed code, or passing tests.

Add regression cases for unequal input exponents, negative operands, widened products, explicit 32-bit wrap/saturation policies, and conversion to a narrower accumulator. Keep archival expectations and any corrected-contract tests separately labeled. In particular, explain how the Listing 4 product at exponent -5 reaches its `(8,-6)` accumulator. Use explicit wide-integer arithmetic and the chosen narrowing policy in the reference; do not depend on signed-overflow undefined behavior.

Use an independent integer/rational or sufficiently precise reference, not a rephrasing of the same implementation. For mixed-precision GPU matrix instructions, use architecture-specific instruction-level validation rather than assuming IEEE FP32 sequential accumulation.

### Test layers and gates

- **Unit tests:** exhaustive 8-bit operand pairs at a documented set of equal/unequal exponents; boundary and randomized 16/32-bit cases; ties, sign extension, overflow, underflow, and long reductions. This is a proposed coverage target, not completed coverage.
- **ABI/dispatch tests:** malformed names/types fail clearly; supported calls reach intended handlers; source→IR→PTX→runtime examples preserve width/exponent metadata. Expected failures must not be counted as arithmetic support.
- **Kernel tests:** matrix outputs match the numerical contract bit-for-bit where exactness is intended; report mismatch count and maximum absolute/relative error otherwise. Test non-square/unaligned shapes and boundary dimensions, including Listing 4's output initialization before its bounds guard.
- **Model tests:** quantify layerwise error, clipping/overflow rates, output agreement, and task accuracy on the same checkpoint and input set.

Keep known archival behavior in regression tests even when introducing corrected semantics. A CI test pass does not establish model accuracy, timing accuracy, or energy accuracy.

## 4. Bounded benchmark suite

Start small enough to explain every result; broaden only after correctness gates pass.

| Tier | Workloads | Purpose and constraints |
|---|---|---|
| A: arithmetic and memory | Conversion, add/mul, dot product/reduction, copy/pack/unpack; dependent chains and independent streams | Separate latency from throughput; isolate alignment, conversion, and memory effects. Include negative and outlier-heavy values. |
| B: dense kernels | GEMM/GEMV with square and rectangular shapes; original 784×100 and 100×10 weight shapes; batch 1 and larger batches | Sweep compute intensity and accumulator length. Explicitly distinguish naive/tiled/library kernels and float-buffer/packed-buffer variants. |
| C: original application | Recover the paper's MNIST checkpoint/network; evaluate all 10,000 test images for the new evaluation | Preserve the one-epoch model only for historical reconstruction. A better-trained checkpoint is a separate experiment, not a replacement baseline. |
| D: modern application | A pretrained ResNet-18 or MobileNetV2 with fixed ImageNet evaluation/preprocessing; optionally a BERT-base task | Add convolution/depthwise and Transformer behavior without beginning with full LLM simulation. Lock dataset, checkpoint, preprocessing, and calibration set. Hardware/tool availability determines final choice. |
| E: optional extension | One small decoder model or representative Transformer block; later, sparsity | Separate prefill/decode, shapes, token lengths, and kernel-versus-model scope. Trace sampling/extrapolation must itself be validated. Sparsity is a separate study with metadata and packing costs. |

Use CUTLASS/cuBLAS/cuDNN only where the chosen hardware, datatype, and simulation/tracing path are actually supported. Rodinia or DeepBench can help regression coverage but should not be the sole evidence of modern AI relevance. Do not claim entire model support based on a single GEMM or transformer layer.

## 5. Baselines and ablations

### Required baseline matrix

- **FP32:** equivalent original algorithm, plus an optimized hardware/library baseline when claiming practical performance. Record TF32/FMA settings rather than assuming “FP32” implies identical arithmetic.
- **Archival fixed point:** preserve original widths/exponents and conversion placement for the reproduction track.
- **Corrected/calibrated fixed point:** separate weights, activation, and accumulator widths; compare fixed power-of-two scales with calibrated per-tensor/per-channel scales. Never calibrate on the test set.
- **INT8 W8A8 / wider accumulation:** compare ordinary integer lowering and available optimized kernels. A scalar integer operation is not a Tensor Core operation.
- **FP16/BF16:** only on supported targets, with specified accumulation. Add **FP8 E4M3/E5M2** as a later comparison when a credible numerical and timing path exists; a float-emulated cast does not establish native FP8 throughput.
- **No custom instruction baseline:** integer/shift implementation of the same mathematical contract, including all conversions and memory costs. This isolates the value of a proposed ISA extension.

### Ablations that answer the actual question

1. Change arithmetic precision while holding global float storage fixed.
2. Change tensor storage/packing while holding the arithmetic policy fixed.
3. Move conversion outside the inner loop versus inside it; include any extra preprocessing and buffers.
4. Compare narrow accumulation with wider accumulation and output-only requantization.
5. Compare manual scales with calibrated scales at a fixed accuracy-loss budget.
6. Sweep assumed custom-unit latency, throughput, and energy independently; report where the conclusion reverses.

The primary plot should be an **accuracy–latency–energy Pareto frontier**, not an energy bar chart that hides a 16-percentage-point accuracy drop. Predeclare task-specific acceptable accuracy loss with the project owner; do not invent an acceptable threshold after observing results.

## 6. Timing, energy, and statistical safeguards

### Timing / modern migration

A low-risk path is to first lower fixed-point semantics to existing integer instructions and validate numerical outputs. Then profile supported CUDA binaries and simulate their real SASS stream with a pinned, supported Accel-Sim configuration. A trace for existing code does **not** automatically model a hypothetical custom instruction: different arithmetic may change values, branches, addresses, packing, and register pressure. Such experiments need a suitable execution-driven path or justified trace regeneration/augmentation and an explicit hypothetical execution model.

Validate against hardware at several workload sizes using latency plus instruction counts, memory transactions, cache behavior, and occupancy where available. Report per-kernel errors and outliers as well as correlation; a high correlation can coexist with systematic bias. Unsupported instructions must not silently become NOPs or convenient lower-cost equivalents.

### Energy

For real hardware, integrate measured power over the relevant interval and specify whether the result is device-only, board, or whole system. Warm up, repeat short kernels sufficiently, report clocks/temperature/power limits, measurement sampling and idle subtraction policy. Include conversion, packing, transfer, and preprocessing in an end-to-end result; show kernel-only numbers separately.

For simulation, separate dynamic and static energy and specify units, coefficients, activity mapping, duration, and calibration target. AccelWattch is a relevant framework, but its published validation does not validate a new functional unit or every new architecture. Horowitz-style per-operation estimates are useful sensitivity inputs, not calibrated GPU board energy. Without hardware/synthesis calibration, report **model-estimated energy** and coefficient sensitivity, not measured savings.

### Accuracy / uncertainty

Use identical held-out examples for paired format comparisons; keep calibration data separate. Report sample count and binomial confidence intervals for classification, and paired prediction disagreement where useful. Fixed checkpoint quantization does not require invented “training seeds”; report multiple seeds when retraining or stochastic calibration is actually used. For performance and power measurements, publish repeated-run distributions/intervals and exclusions. Deterministic simulation repeats do not substitute for model-error validation.

## 7. Release gates and artifact checklist

| Gate | Minimum exit evidence |
|---|---|
| G0: provenance | Fixed simulator commit; recover/identify upstream base; reconcile version strings; archive compiler/frontend revisions, model weights, configuration, and missing-artifact list. |
| G1: semantics | Numerical contract; reference implementation; pass/fail coverage report; source–paper discrepancies resolved or documented. |
| G2: reproducibility | Fresh build and a source-to-runtime example with logs under an explicitly tested toolchain; maintenance README/test docs updated. |
| G3: historical audit | Reproduced or reconstructed Figures 4–6 with full scope labels; memory-traffic explanation established or withdrawn. |
| G4: modern baseline | Verified supported instruction/tool/config path; numerical results plus hardware timing correlation. |
| G5: new research claim | Accuracy-constrained comparisons, ablations, raw logs and scripts, sensitivity analysis; energy measurement/model clearly distinguished. |

Store a machine-readable run manifest with commits, CUDA/driver/compiler/library versions, GPU model/configuration, flags, shapes/batch sizes, checkpoint/data hashes, seeds, timing scope, measurement protocol, and result status. Archive generated PTX/SASS, counters, numerical mismatch reports, and figure-generation inputs. A failed build or unsupported case is useful evidence and must remain visible.

### Inputs still needed from the project owner

- Matching original LLVM/Clang/SPIR-V/device-compiler source and build instructions; original experimental scripts/configurations and generated PTX.
- Original MNIST checkpoint, training settings, evaluated image indices, and raw Figures 4–6 data; actual global-memory layout used in those runs.
- Confirm and record the intended multiplication contract: the relayed direction favors raw product/exponent addition, supported by archived test expectations, but is not independently verified owner approval or implementation. Resolve Algorithm 2 width-sum versus bounded/wrapped test outputs, destination conversion, overflow, rounding, and accumulation policies.
- Available GPUs, CUDA/driver versions, compute/storage budget, profiling access, and power-measurement capability.
- Whether the immediate target is archival reproduction, trustworthy numerical infrastructure, or a modern performance/energy study; acceptable task-accuracy loss.
- Which maintenance PR commit and test logs may be cited. Do not advertise planned code changes as merged or validated.
