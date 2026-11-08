# Fixed-point GPU simulation: research context and evidence boundaries

## Status and scope

This document connects the original research to the archived implementation and a proposed modernization effort. It is **not a report of a completed CUDA upgrade, Accel-Sim port, successful build, or new experiment**.

Source inspection is pinned to [`3107bc1369c6ef09d79ec36fc04eae494f24611d`](https://github.com/jaspercllee/GPGPU-Sim-Fxp/tree/3107bc1369c6ef09d79ec36fc04eae494f24611d). Later maintenance commits must be assessed separately. “Present in source” below does not mean tested or correct. No build or experiment was run for this documentation review.

## Maintainer note for this branch

The tables below are the review of snapshot `3107bc1369c6ef09d79ec36fc04eae494f24611d`. Commit `4fe5067` moved the helpers and locked the old float-scaling multiply. A later commit on this branch changes `binary_mul` to a raw product plus exponent sum, and adds `binary_sub` and `binary_div`. Division is not in the paper algorithm. CPU tests cover those helpers. They are not a rerun of the MNIST figures. Original weights and scripts were reported unavailable. A fresh-weight MLP proposal, separate from any CNN, is [RECONSTRUCTION_PLAN.md](RECONSTRUCTION_PLAN.md). The historical paper claims in this file stay attributed review, not new measurements.


### Abstract-style project summary

GPGPU-Sim-Fxp explores compiler-visible fixed-point arithmetic in an execution-driven GPU simulation environment. The 2020 paper describes an OpenCL/LLVM/SPIR-V-to-PTX workflow, fixed-point representations parameterized by width and binary exponent, and functional support for conversions and arithmetic. Its evaluation used an MNIST inference workload and a Fermi-like GPU model with GPUWattch-based energy estimates. The paper reported a trade-off between modeled energy and classification accuracy; these historical results are neither measurements of current GPUs nor evidence of a modern low-precision execution unit. The inspected implementation handles fixed-point operations through recognized PTX function calls. Current research priorities are reproducible numerical semantics, explicit accounting for conversion and memory-layout costs, and separate validation of functional correctness, timing, and energy. Modern CUDA and Accel-Sim integration remain proposed work pending implementation and validation.

## Original publication

Chao-Lin Lee, Min-Yih Hsu, Bing-Sung Lu, Ming-Yu Hung, and Jenq-Kuen Lee. **Experiment and enabled flow for GPGPU-Sim simulators with fixed-point instructions.** *Journal of Systems Architecture* **111**, article 101783, 2020. [DOI: 10.1016/j.sysarc.2020.101783](https://doi.org/10.1016/j.sysarc.2020.101783).

The seven-page publisher PDF was read in full for this review; page references below count PDF pages, starting with the title page. The supplied original LaTeX (`JSA2019.tex`) and its figure PDFs were subsequently cross-checked read-only: Algorithm 2, Listings 3–4, and Figures 4–6 agree with the corresponding publisher-PDF content discussed here. This cross-check does not establish which executable produced the figures. The [PLLab publication list](https://pllab.cs.nthu.edu.tw/~jklee/papers.html) also identifies the earlier 2018 workshop paper, *Enable the Flow for GPGPU-Sim Simulators with Fixed-Point Instructions*, [DOI: 10.1145/3229710.3229722](https://doi.org/10.1145/3229710.3229722). Do not describe the 2020 article as unpublished or fixed-point arithmetic itself as a newly invented number format.

### Historical fixed-point workflow

The paper's Sections 3.1–3.3 (pp. 3–4) describe:

1. OpenCL kernel source is compiled by Clang to LLVM IR.
2. A modified LLVM/SPIR-V flow expresses fixed-point conversions and operations.
3. SPIR-V is translated back to LLVM IR; the LLVM backend produces PTX.
4. A modified GPGPU-Sim functional model executes fixed-point operations.
5. Timing activity feeds the GPUWattch-based energy model.

This is a **paper-described workflow**, not a verified end-to-end build recipe for this snapshot. The inspected repository establishes the simulator-side call dispatch; it does not by itself establish availability of the matching historical producer compiler. The fixed-point SPIR-V extension described in the paper should not be represented as a ratified standard.

Section 3.2 defines values as `N * 2^E`, using a two's-complement signed representation. Width and exponent determine range and resolution. Distinguish the following, which are not interchangeable:

- fixed-point arithmetic and its overflow/rounding semantics;
- affine integer quantization, usually `x ≈ s * (q - z)`;
- packed integer instructions and INT8 matrix instructions;
- FP8 formats and architecture-specific Tensor Core arithmetic.

## Capability and limitation matrix

All anchors in this table refer to the pinned snapshot, not a moving branch.

| Area | Observed evidence | Boundary / required follow-up |
|---|---|---|
| Base version | [`version`](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/version#L1-L2) identifies 3.2.2; [`CHANGES`](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/CHANGES#L1-L22) begins with 3.2.3+edits. | Preserve this ambiguity. Neither string identifies the exact upstream base commit. Do not relabel it GPGPU-Sim 4.x or Accel-Sim. |
| Historical CUDA context | `CHANGES` describes initial support for CUDA 5.0/5.5/6.0/7.5; the [tree](https://github.com/jaspercllee/GPGPU-Sim-Fxp/tree/3107bc1369c6ef09d79ec36fc04eae494f24611d/build/gcc-4.8.4) contains CUDA-7000/7050 build directories. | These are archival provenance clues, not proof that checked-in binaries rebuild or that current CUDA is supported. |
| Fixed-point dispatch | [`instructions.cc`, `call_impl`](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/src/cuda-sim/instructions.cc#L1468-L1535) recognizes fixed-point binary/conversion/math function names and invokes SPIR-V runtime handlers. | Simulator intrinsics encoded as PTX calls; not evidence of a native NVIDIA fixed-point opcode or hardware functional unit. |
| Widths and binary operations | [`spirv_device_runtime.cc`, binary dispatch](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/src/cuda-sim/spirv_device_runtime.cc#L780-L860) contains 8/16/32-bit handling and dispatches add/mul. | Recognized names are not all implemented operations. Numerical behavior remains unvalidated. |
| Subtraction and division | [Name parser](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/src/cuda-sim/spirv_fixed_point.cc#L21-L39) recognizes add/sub/mul/div, but the runtime dispatch above selects only add/mul and otherwise asserts. | Paper Section 3.3 / Algorithm 2 includes subtraction. Document this source–paper discrepancy rather than advertising subtraction support. |
| Conversion and rounding | [Conversion parser](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/src/cuda-sim/spirv_fixed_point.cc#L41-L96) recognizes RTE/RTN/RTZ/RTP; [runtime conversion functions](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/src/cuda-sim/spirv_device_runtime.cc#L255-L664) implement conversion paths. | Four named modes are source-visible, not a conformance result. The parser includes `TODO: Saturation`; overflow and special values need explicit contracts and tests. |
| Transcendentals | [`gpgpusim_spirv_invokeFxpMathFunction`](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/src/cuda-sim/spirv_device_runtime.cc#L862-L887) handles exp/log through float callbacks. | Not an integer-only fixed-point exp/log hardware model. Do not infer low-power transcendental execution. |
| Multiplication semantics | [`binary_mul`](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/src/cuda-sim/spirv_device_runtime.cc#L158-L209) sets `ret_exp = lhs_exp`, uses float `pow`, and scales an integer product. | This differs from Algorithm 2 and the archived gtest raw-product/exponent-sum expectations below. Keep archival implementation, test expectations, and proposed semantics separate; no updated implementation or passing logs were inspected. |
| Tests | [`tests/spirv`](https://github.com/jaspercllee/GPGPU-Sim-Fxp/tree/3107bc1369c6ef09d79ec36fc04eae494f24611d/tests/spirv) contains arithmetic, conversion, and math tests. | The arithmetic gtest assertions were also read: they expect a raw product and exponent addition, including a wrapped 32-bit case. These are expected outputs, not passing results. Parser-only sub/div tests do not establish executable arithmetic support. New maintenance test results must cite their own commit and logs. |
| Timing and energy | Paper Section 3.4 maps conversion cost to shifts and binary arithmetic cost to integer operations. | This is a modeling assumption. The inspected dispatch does not establish dedicated-unit latency, modern INT8/FP8 throughput, or validated power coefficients. |
| Modern integration | Accel-Sim and newer CUDA are relevant modernization targets. | No completed port, Tensor Core support, modern GPU validation, or new end-to-end compiler pipeline was established by this review. A migration plan or CI configuration alone does not establish these capabilities. |

### Multiplication: four distinct evidence levels

| Evidence level | What is established | What is not established |
|---|---|---|
| Paper Algorithm 2 | The LaTeX multiplication branch assigns `E_out = E_lhs + E_rhs`, `W_out = W_lhs + W_rhs`, and `q_out = q_lhs * q_rhs` before any later conversion. | It does not completely specify finite-width overflow/narrowing or how a widened product is assigned to a narrower accumulator. |
| Archived gtest expectations | [`fixed_point_arithmetic_impl.cc`](https://github.com/jaspercllee/GPGPU-Sim-Fxp/blob/3107bc1369c6ef09d79ec36fc04eae494f24611d/tests/spirv/fixed_point_arithmetic_impl.cc), `BinaryMulBasic`, expects raw operands 7 and 8 at exponents -2/-2 to produce raw 56 at exponent -4. `BinaryMulDiffWidthExp` expects exponent -5 and raw `4293885953`, explicitly described as wrapped overflow, with a 4-byte return. | The tests were not run. These assertions support raw-product/exponent-addition intent, but do **not** match the paper's unrestricted width-sum rule in all cases: the tests use 32-bit output even for 32-bit input operands. Test names/comments are not a complete width contract. |
| Archived runtime implementation | The pinned `binary_mul` uses `ret_exp = lhs_exp` and float-based scaling of an integer product. | This is not the same raw-result contract as the assertions. Source inspection alone supplies no pass/fail log or explanation of historical experimental behavior. |
| Proposed contract / unverified new code | Raw integer multiplication with exponent addition is a proposed contract consistent with the paper's multiplication branch and these test expectations on raw value/exponent. | A relayed report of a design direction is not independently verified owner approval, implementation, or test success. No newer implementation was inspected. Output width, overflow, narrowing, and accumulation still require an explicit policy and evidence. |

For the proposed contract, multiplication itself should not silently rescale the raw product; any conversion to a destination exponent/width is a separate, specified step. A reference may use a wide integer product followed by explicit modulo or saturation **only under a documented output policy**. A test comment about wrapping does not justify relying on C++ signed-overflow undefined behavior. Do not call the proposal fully paper-equivalent until the width-sum versus bounded-output distinction is resolved.

### Build and use

Use the maintained [README](../README.md) and the maintenance PR's test/build documentation as the operational source of truth. The older [README](../README) and existing [SPIR-V test Makefile](../tests/spirv/Makefile) provide historical context, not a promise of compatibility. This research document intentionally supplies no unverified install/build/run commands. Record tested host compiler, CUDA toolkit/driver, simulator commit, and frontend/compiler revision before publishing a compatibility claim. Do not treat checked-in build objects as reproducible artifacts.

## What the 2020 results actually establish

| Paper location | Historical observation | Correct interpretation |
|---|---|---|
| Section 4.1, p. 5; Listing 3 | MNIST network with 784→100→10 matmuls and sigmoid; one training epoch; reported FP baseline accuracy 90%. | A small proof-of-concept inference workload, not validation across CNNs, Transformers, or production inference stacks. |
| Table 2, p. 6 | Fermi-like configuration, 16 SMs; per-operation energy values attributed to Horowitz (ISSCC 2014). | Historical model and assumed operation costs, not measured energy on a current GPU. Table entries in pJ are **energy per operation**, not power. |
| Figure 5 / Section 4.2, p. 6 | Normalized total modeled GPU energy: FP32 100%, FXP32 92%, FXP8 86% (rounded figure labels). Source figure absolute values: 48,489,634,234.63 / 44,748,439,820.38 / 41,702,835,684.92 pJ. | Reported 8% and 14% reductions for the evaluated configurations only. The abstract's “at least 14%” overstates the scope; 14% is not a general lower bound. |
| Figure 6 / Section 4.2, p. 6 | FP32 90%, FXP(32,-16) 82%, FXP(8,-4) 74%; text describes 50 recognition images. | An 8- and 16-percentage-point accuracy decrease, not “limited” or negligible loss. The text does not clearly reconcile the 10,000-image test set in Section 4.1 with the 50-image evaluation. |
| Figure 4, p. 6 | FXP8 DRAM energy normalized to 85% of FXP32 (rounded label); source figure absolute values: FXP32 129,283,435.82 pJ, FXP8 109,984,014.63 pJ. | A modeled observation requiring a memory-traffic audit, not automatic evidence of 8-bit tensor storage. |

**Important internal consistency issue:** Listing 4 (p. 5) declares global A/B/C buffers as `float*`, loads float values, converts inside the loop, and writes float output. Therefore narrower arithmetic alone does not demonstrate smaller off-chip tensors. The explanation on p. 6 that narrower representation caused reduced DRAM energy needs generated PTX, buffer layouts, transaction counts, cycles, and power-model evidence. Another implementation may have been used, but it was not established by the displayed kernel. Do not carry this explanation forward as a verified causal result.

**Listing-versus-figure precision settings:** the LaTeX Listing 4 uses `A_fxp = fixed_point<8,-2>`, `B_fxp = fixed_point<8,-3>`, and an accumulator `fixed_point<8,-6>`. Figure 6 instead labels the FXP8 configuration `(8,-4)`. The listing may be illustrative; it does not establish the exact configuration that generated the figure. Under the proposed raw-product/exponent-addition rule, the input product has exponent -5 and nominal full width 16, so assignment/accumulation at `(8,-6)` needs an explicit conversion/narrowing contract. Recover the actual experimental kernel and compiler lowering before equating these configurations. Listing 4 also writes `C[i*Mdim + j] = 0` before its bounds guard; boundary-launch tests should audit this separately.

The article discusses performance improvements, but Section 4 presents energy/accuracy plots rather than an explicit latency/speedup evaluation. Do not supply a speedup number or infer one from energy reduction. Also remove the unconditional statement in Section 5 that fixed point is more accurate than BF16: representable range, scale, rounding, accumulation, and workload all matter.

## What may be updated now, and what changed since 2020

**Safe editorial updates now:** clarify the representation and simulator-call mechanism; label historical results; fix the “at least 14%” and accuracy descriptions; separate energy from power; add source–paper discrepancies and modern related work; state missing toolchain and validation evidence.

**Not established implementation changes:** modern CUDA compatibility, Accel-Sim integration, new benchmarks, improved accuracy, new energy estimates, and faster execution. Keep these under “proposed” until corresponding code and experiments exist. The present maintenance scope communicated during review is baseline-preserving tests/CI/docs plus a phased migration plan, not a complete port; merge status and test results must be supplied by maintainers.

See [the experiment and rerun plan](experiment-rerun-plan.md) before replacing historical figures or making a new scientific claim.

## Verified related work: why the research framing must change

These references were checked against publisher, author, or arXiv pages. They are contextual references, not evidence that this repository implements their features.

1. **Khairy, Shen, Aamodt, and Rogers. “Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling.” ISCA 2020, pp. 473–486.** [DOI](https://doi.org/10.1109/ISCA45697.2020.00047), [author page](https://engineering.purdue.edu/tgrogers/publication/khairy-isca-2020/). Contemporary with the original paper, not post-2020. Combines SASS trace-driven simulation, an updated GPGPU-Sim model, and validation infrastructure. Distinguish functional execution of hypothetical arithmetic from timing replay of existing machine instructions.
2. **Kandiah et al. “AccelWattch: A Power Modeling Framework for Modern GPUs.” MICRO 2021, pp. 738–753.** [DOI](https://doi.org/10.1145/3466752.3480063), DOI above. Models constant/static/dynamic power and is validated on Volta. It is a relevant replacement/comparison framework, not automatic validation of a custom fixed-point unit or Hopper power.
3. **Nagel, Fournarakis, Amjad, Bondarenko, van Baalen, and Blankevoort. “A White Paper on Neural Network Quantization.” 2021, arXiv:2106.08295.** [Paper](https://arxiv.org/abs/2106.08295). PTQ/QAT, calibration, and hardware-aware quantization should replace a simplistic float-versus-fixed comparison. This is a white paper/preprint, not a claimed conference publication.
4. **Fasi, Higham, Mikaitis, and Pranesh. “Numerical behavior of NVIDIA tensor cores.” PeerJ Computer Science 7:e330, 2021.** [DOI](https://doi.org/10.7717/peerj-cs.330), [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7959640/). Demonstrates why accumulation/rounding behavior requires targeted tests rather than an idealized mathematical dot product. Later work refines earlier numerical models; do not treat every inferred hardware detail as universal.
5. **Micikevicius et al. “FP8 Formats for Deep Learning.” 2022, arXiv:2209.05433.** [Paper](https://arxiv.org/abs/2209.05433). E4M3/E5M2, scaling, saturation, and wider accumulation are distinct from fixed-point semantics. Its simulated FP8 numerical results are not cycle-level GPU simulation results.
6. **Xiao, Lin, Seznec, Wu, Demouth, and Han. “SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.” ICML 2023, PMLR 202, pp. 38087–38099.** [Publisher](https://proceedings.mlr.press/v202/xiao23c.html). A credible W8A8 calibration baseline for an optional Transformer extension. Its reported gains belong to that work, not this repository.
7. **Xie, Xu, Wang, Yang, and Yang. “Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units: Demystifying Numerical Discrepancy and Accuracy.” arXiv:2511.10909, v3, September 2026.** [Paper](https://arxiv.org/abs/2511.10909). Earlier version titled *MMA-Sim: Bit-Accurate Reference Model of Tensor Cores and Matrix Cores*. Separates instruction-level arithmetic behavior from timing and studies FP8 and other floating-point formats. Cite the version used; do not invent a venue or assume it validates integer fixed-point arithmetic.
8. **Pan et al. “Architecting the Next Generation of Asynchronous, Distributed GPUs for the AI Era.” 2026, arXiv:2608.22602.** [Paper](https://arxiv.org/abs/2608.22602), [Accel-Sim project](https://accel-sim.github.io/). The project announces Accel-Sim 2.0 and Hopper-era asynchronous modeling. This makes a 2026 plan based only on Accel-Sim 1.x incomplete. Select and pin a suitable release; assess actual instruction/tracer/config coverage. The project's broad support statements are not independent verification or guarantees for this repository. Do not copy its correlation numbers into this project's evaluation.

### Candidate research contribution, not an implemented claim

A stronger next step is **semantics-validated, accuracy-constrained precision selection with explicit conversion/packing cost accounting**, rather than “rerun an old simulator on a new benchmark.” Test when a fixed binary scale loses to calibrated INT8, wider accumulation, or FP8, and why. Compare ordinary integer lowering with hypothetical instructions without granting the latter free conversion, packing, or execution throughput. Modern simulation infrastructure is a means of testing that question, not novelty by itself.
