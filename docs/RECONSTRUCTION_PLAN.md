# Fresh-weight reconstruction

The MLP stage has been run on CPU. It is a new one-epoch training, not a reproduction of the 2020 checkpoint, and it does not execute GPGPU-Sim. Numbers are in [results.json](../experiments/mnist_mlp/results.json). The opt-in command is `scripts/run_mnist_mlp.sh`. Dataset archives and the weight blob stay out of git. A conv2d network has not been started.

## What the paper source describes

The supplied paper source (not committed here) sets up an MNIST inference network in NNEF:

- input shape `[1, 784]`
- weights `[784, 100]` and `[100, 10]`
- biases `[1, 100]` and `[1, 10]`
- sigmoid after each `matmul`+`add`
- 60,000 training images, 10,000 test images, one training epoch
- a later accuracy note uses 50 recognition images
- one kernel listing uses `fixed_point<8,-2>`, `fixed_point<8,-3>`, and accumulator `fixed_point<8,-6>`
- the accuracy figure labels FXP32 `(32,-16)` and FXP8 `(8,-4)`

Those two scale settings are different. A reconstruction must not treat the listing and the figure as the same configuration. The source searched here does not describe CIFAR or a conv2d network. A CNN is a new extension, not the paper architecture.

## Track A — paper-shaped MLP, fresh weights

Goal: a new one-epoch MLP with the shapes above, then a CPU numeric comparison. It will not recover the paper's 90/82/74 accuracy figures.

Before any training run, record:

1. Dataset provenance and the exact download. MNIST is not in this tree. Name the archive URL, byte size, and sha256. Do not vendor the images until that checksum is reviewed.
2. Training config: optimizer, learning rate, batch size, loss, one epoch, and one integer seed. Write them in a file that is committed with the script, not only in a shell history.
3. Split: 60,000/10,000 as in the paper source. If a calibration set is added later, cut it from the training 60,000. Do not calibrate on the test 10,000. The paper's 50-image accuracy note is a separate evaluation list; if those indices are unknown, say so and do not invent them.
4. Checkpoint sha256 after training, plus the script hash. No checkpoint exists yet.
5. Training budget: one epoch on CPU or a single GPU is enough for this shape. State the machine only after the run. Do not start that run in the current change.
6. Two numeric paths, kept separate:
   - CPU reference: the same checkpoint evaluated with ordinary floating-point matmul and sigmoid, and again with the `binary_add` / `binary_mul` contracts in `src/cuda-sim/spirv_fxp_arith.cc`.
   - Simulator path: a real PTX/OpenCL kernel through GPGPU-Sim. This needs the missing device compiler, a chosen CUDA toolkit, and a rebuilt `libcudart.so`. None of those are available in the current change, so this path cannot be claimed.

## Track B — conv2d CNN

Not part of Track A. Choose a dataset only when Track A has a CPU reference and a written split. Do not assume CIFAR. Do not describe a CNN result as the paper's MNIST result.

## Blockers

- Original weights, scripts, generated PTX, and the device compiler are not in the repository.
- CUDA toolkit selection, a simulator rebuild, and any end-to-end kernel run are still undone.
- Listing scales and figure scales disagree, and the 50-image index list is not in the tree.
- `exp`/`log` exist as float callbacks; the paper network uses sigmoid, which is not a named `fxp_` operation in the parser. A reconstruction has to say whether sigmoid stays in float outside the fixed-point matmul.
