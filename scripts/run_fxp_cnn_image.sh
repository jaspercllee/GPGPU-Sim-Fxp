#!/bin/sh
# One MNIST test image through the bespoke CNN PTX on GPGPU-Sim.
# Functional simulation, power left off. Not a 10k accuracy run.
# The PTX is produced by experiments/legacy_ptx/gen_fxp_cnn_ptx.py.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
CHROOT=/var/tmp/ubuntu-14.04-cuda75
DATA="$ROOT/experiments/mnist_mlp/data"
WEIGHTS="$ROOT/experiments/mnist_cnn/out/weights.bin"
OUT="$ROOT/experiments/mnist_cnn/out"
INDEX="${1:-0}"
if [ ! -x "$CHROOT/usr/local/cuda-7.5/bin/nvcc" ]; then
  echo "Ubuntu 14.04 CUDA 7.5 chroot is not present" >&2
  exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
  echo "missing $WEIGHTS ; run scripts/run_mnist_cnn.sh first" >&2
  exit 1
fi
mkdir -p "$OUT"
make -C "$ROOT/experiments/mnist_cnn"
python3 "$ROOT/experiments/legacy_ptx/gen_fxp_cnn_ptx.py" \
  "$ROOT/experiments/legacy_ptx/fxp_cnn_image.ptx"
"$ROOT/experiments/mnist_cnn/eval_fxp_cnn" --export-sim \
  "$DATA" "$WEIGHTS" "$OUT/sim_case.bin" "$INDEX"
sudo python3 - << 'PY'
from pathlib import Path
p = Path("/var/tmp/ubuntu-14.04-cuda75/tmp/run/gpgpusim.config")
t = p.read_text()
t = t.replace("-power_simulation_enabled 1", "-power_simulation_enabled 0")
t = t.replace("-power_simulation_enabled                    1",
              "-power_simulation_enabled                    0")
p.write_text(t)
PY
sudo cp "$ROOT/experiments/legacy_ptx/fxp_cnn_image.ptx" "$CHROOT/tmp/run/fxp_cnn_image.ptx"
sudo cp "$ROOT/experiments/legacy_ptx/fxp_cnn_host.cu" "$CHROOT/tmp/run/fxp_cnn_host.cu"
sudo cp "$OUT/sim_case.bin" "$CHROOT/tmp/run/sim_case.bin"
sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /tmp/run; nvcc -cudart shared -arch=sm_20 fxp_cnn_host.cu -o fxp_cnn_host'
# Names the override file must provide.
sudo chroot "$CHROOT" bash -lc 'export PATH=/usr/local/cuda-7.5/bin:$PATH; cuobjdump -ptx /tmp/run/fxp_cnn_host | grep -E "entry" | head' || true
START=$(date +%s)
set +e
sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /tmp/run; PTX_SIM_MODE_FUNC=1 PTX_SIM_USE_PTX_FILE=1 PTX_SIM_KERNELFILE=/tmp/run/fxp_cnn_image.ptx LD_LIBRARY_PATH=/src/lib/gcc-4.8.4/cuda-7050/release ./fxp_cnn_host /tmp/run/sim_case.bin' > "$OUT/sim_cnn.log" 2>&1
STATUS=$?
set -e
END=$(date +%s)
echo "HOST_EXIT $STATUS WALL_SEC $((END - START))" | tee -a "$OUT/sim_cnn.log"
python3 - "$OUT/sim_cnn.log" "$STATUS" "$((END - START))" "$INDEX" "$ROOT/experiments/legacy_ptx/cnn_one_image.json" << 'PY'
import json, sys
log, status, wall, index, out = sys.argv[1:]
text = open(log, errors="replace").read()
def between(name):
    a = text.find("SIMCNN LAYER %s begin" % name)
    b = text.find("SIMCNN LAYER %s end" % name)
    if a < 0 or b < 0:
        return None
    return text[a:b].count("1519: call_impl_in_binary:")
layers = ["probe", "requant", "tile", "conv", "pool", "fc"]
counts = {k: between(k) for k in layers}
import re
def field(name):
    m = re.search(r"SIMCNN %s ([^\n]*)" % name, text)
    return m.group(1).strip() if m else ""
def ints(s):
    return [int(x) for x in s.split()] if s else []
layer_seconds = {}
for k in layers:
    m = re.search(r"SIMCNN LAYER %s end seconds ([0-9.]+)" % k, text)
    layer_seconds[k] = float(m.group(1)) if m else None
record = {
    "producer": "bespoke loop PTX, not the 2020 LLVM/SPIR-V compiler",
    "image_index": int(index),
    "host_exit": int(status),
    "wall_sec": int(wall),
    "functional_mode": "1=functional simulation only" in text,
    "power_line": next((ln.strip() for ln in text.splitlines() if "power_simulation_enabled" in ln), ""),
    "call_impl_between_markers": counts,
    "call_impl_total": text.count("1519: call_impl_in_binary:"),
    "layer_seconds": layer_seconds,
    "probe_mismatch": field("probe_mismatch"),
    "requant_mismatch": field("requant_mismatch"),
    "tile": field("tile got"),
    "conv_mismatch": field("conv_mismatch"),
    "pool_mismatch": field("pool_mismatch"),
    "logit_mismatch": field("logit_mismatch"),
    "sim_logits": ints(field("sim_logits")),
    "cpu_fxp_logits": ints(field("cpu_fxp_logits")),
    "summary": field("sim_argmax"),
    "note": "One functional image. CPU float argmax is from the host export, not a simulator float kernel. Not a 10k accuracy claim and not an energy result.",
}
open(out, "w").write(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
sys.exit(0 if int(status) == 0 else 1)
PY
