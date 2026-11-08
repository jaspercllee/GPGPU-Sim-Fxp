#!/bin/sh
# Timing-mode GPUWattch run of one MNIST image on the bespoke CNN PTX.
# Not functional mode. Not an energy comparison against float.
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
make -C "$ROOT/tests/cpu_fxp"
make -C "$ROOT/experiments/mnist_cnn"
python3 "$ROOT/experiments/legacy_ptx/gen_fxp_cnn_ptx.py" \
  "$ROOT/experiments/legacy_ptx/fxp_cnn_image.ptx"
"$ROOT/experiments/mnist_cnn/eval_fxp_cnn" --export-sim \
  "$DATA" "$WEIGHTS" "$OUT/sim_case.bin" "$INDEX"
sudo cp "$ROOT/src/gpuwattch/XML_Parse.h" "$ROOT/src/gpuwattch/XML_Parse.cc" \
  "$ROOT/src/gpuwattch/fxp_power_policy.h" "$ROOT/src/gpuwattch/gpgpu_sim_wrapper.cc" \
  "$ROOT/src/gpuwattch/core.cc" /var/tmp/gpgpu-src/src/gpuwattch/
sudo cp "$ROOT/src/gpgpu-sim/shader.cc" "$ROOT/src/gpgpu-sim/gpu-sim.cc" \
  "$ROOT/src/gpgpu-sim/shader.h" /var/tmp/gpgpu-src/src/gpgpu-sim/
sudo cp "$ROOT/src/abstract_hardware_model.h" /var/tmp/gpgpu-src/src/
sudo cp "$ROOT/src/cuda-sim/ptx_ir.cc" "$ROOT/src/cuda-sim/cuda-sim.cc" \
  "$ROOT/src/cuda-sim/cuda-sim.h" /var/tmp/gpgpu-src/src/cuda-sim/
sudo cp "$ROOT/configs/GTX480/gpuwattch_gtx480.xml" "$CHROOT/tmp/run/gpuwattch_gtx480.xml"
sudo python3 - << 'PY'
from pathlib import Path
p = Path("/var/tmp/ubuntu-14.04-cuda75/tmp/run/gpgpusim.config")
lines = []
for ln in p.read_text().splitlines():
    if ln.startswith("-power_simulation_enabled"):
        ln = "-power_simulation_enabled 1"
    elif ln.startswith("-gpgpu_runtime_stat"):
        # McPAT is invoked once per sample. 10000 cycles keeps one image bounded.
        ln = "-gpgpu_runtime_stat 10000"
    lines.append(ln)
p.write_text("\n".join(lines) + "\n")
PY
sudo cp "$ROOT/experiments/legacy_ptx/fxp_cnn_image.ptx" "$CHROOT/tmp/run/fxp_cnn_image.ptx"
sudo cp "$ROOT/experiments/legacy_ptx/fxp_cnn_host.cu" "$CHROOT/tmp/run/fxp_cnn_host.cu"
sudo cp "$OUT/sim_case.bin" "$CHROOT/tmp/run/sim_case.bin"
# gpu-sim.cc is compiled only when sources change. The power macro is a
# compiler flag, so a previous power-off object must be rebuilt.
# Instruction layout must match every object that includes the hardware header.
sudo rm -f /var/tmp/gpgpu-src/build/gcc-4.8.4/cuda-7050/release/gpgpu-sim/*.o \
  /var/tmp/gpgpu-src/build/gcc-4.8.4/cuda-7050/release/cuda-sim/*.o \
  /var/tmp/gpgpu-src/build/gcc-4.8.4/cuda-7050/release/*.o
sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /src; source setup_environment; make -j2'
sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /tmp/run; nvcc -cudart shared -arch=sm_20 fxp_cnn_host.cu -o fxp_cnn_host'
# One binary, four uncalibrated scales. 0 is the no-compute-proxy control.
SUMMARY="$ROOT/experiments/legacy_ptx/cnn_one_image_power.json"
: > "$OUT/power_scale_index.txt"
for SCALE in 0 0.5 1 2; do
  TAG=$(printf '%s' "$SCALE" | tr '.' 'p')
  sudo python3 - "$SCALE" << 'PY'
import sys
from pathlib import Path
scale = sys.argv[1]
p = Path("/var/tmp/ubuntu-14.04-cuda75/tmp/run/gpgpusim.config")
lines = [ln for ln in p.read_text().splitlines() if not ln.startswith("-fxp_compute_proxy_scale")]
lines.append("-fxp_compute_proxy_scale %s" % scale)
p.write_text("\n".join(lines) + "\n")
PY
  sudo rm -f "$CHROOT/tmp/run"/gpgpusim_power_report__*.log
  START=$(date +%s)
  set +e
  sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /tmp/run; PTX_SIM_USE_PTX_FILE=1 PTX_SIM_KERNELFILE=/tmp/run/fxp_cnn_image.ptx LD_LIBRARY_PATH=/src/lib/gcc-4.8.4/cuda-7050/release ./fxp_cnn_host /tmp/run/sim_case.bin' > "$OUT/sim_cnn_power_s${TAG}.log" 2>&1
  STATUS=$?
  set -e
  END=$(date +%s)
  echo "HOST_EXIT $STATUS WALL_SEC $((END - START)) SCALE $SCALE" | tee -a "$OUT/sim_cnn_power_s${TAG}.log"
  REPORT=$(ls -1t "$CHROOT/tmp/run"/gpgpusim_power_report__*.log 2>/dev/null | head -1 || true)
  if [ -n "$REPORT" ]; then
    sudo cp "$REPORT" "$OUT/power_report_s${TAG}.log"
    sudo chown "$(id -u):$(id -g)" "$OUT/power_report_s${TAG}.log" || true
  fi
  echo "$SCALE $STATUS $((END - START)) $OUT/sim_cnn_power_s${TAG}.log $OUT/power_report_s${TAG}.log" >> "$OUT/power_scale_index.txt"
  if [ "$STATUS" -ne 0 ]; then
    echo "scale $SCALE failed" >&2
    exit "$STATUS"
  fi
done
python3 - "$OUT/power_scale_index.txt" "$SUMMARY" "$INDEX" << 'PY'
import json, re, sys
index_path, out, image_index = sys.argv[1:]
def blocks(src):
    parts = src.split("kernel_name = ")
    out = []
    for part in parts[1:]:
        name = part.split("\n", 1)[0].strip()
        if name:
            out.append((name, part))
    return out
def grab(block, pat):
    m = re.search(pat, block)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
def field(text, name):
    m = re.search(r"SIMCNN %s ([^\n]*)" % name, text)
    return m.group(1).strip() if m else ""
def ints(s):
    return [int(x) for x in s.split()] if s else []
def last_component(report, label):
    vals = re.findall(r"%s\|[^|\n]*\|[ \t]*([0-9]+(?:\.[0-9]+)?)" % label, report)
    return float(vals[-1]) if vals else None
runs = []
for line in open(index_path):
    scale_s, status_s, wall_s, simlog, report = line.split()
    text = open(simlog, errors="replace").read()
    rep = open(report, errors="replace").read() if report else ""
    named = []
    for name, block in blocks(rep):
        named.append(name)
    cycles = [int(x) for x in re.findall(r"gpu_tot_sim_cycle = (\d+)", text)]
    events = re.findall(r"fxp_proxy_events add=(\d+) sub=(\d+) mul=(\d+) div=(\d+) cvt=(\d+)", text)
    runs.append({
        "scale": float(scale_s),
        "host_exit": int(status_s),
        "host_wall_sec": int(wall_s),
        "functional_mode": "mode=functional simulation" in text,
        "printed_scale": (re.findall(r"fxp_compute_proxy_scale = ([0-9.eE+-]+)", text) or [None])[-1],
        "last_gpu_tot_sim_cycle": cycles[-1] if cycles else None,
        "modeled_time_sec": (cycles[-1] / 700e6) if cycles else None,
        "run_cumulative_gpu_pJ": last_component(rep, "GPU"),
        "run_cumulative_gpu_J": (last_component(rep, "GPU") or 0) * 1e-12,
        "cumulative_const_dynamic_pJ": last_component(rep, "CONST_DYNAMIC"),
        "cumulative_sp_alu_pJ": last_component(rep, "SP"),
        "cumulative_sfu_mul_pJ": last_component(rep, "SFU"),
        "cumulative_fpu_pJ": last_component(rep, "FPU"),
        "fxp_events_last": [int(x) for x in events[-1]] if events else None,
        "sim_logits": ints(field(text, "sim_logits")),
        "logit_mismatch": field(text, "logit_mismatch"),
        "named_kernel_prints": named,
    })
base = runs[0]["run_cumulative_gpu_J"] if runs else None
for row in runs:
    row["proxy_delta_J_vs_scale0"] = row["run_cumulative_gpu_J"] - base
    row["sp_delta_pJ_vs_scale0"] = (row["cumulative_sp_alu_pJ"] or 0) - (runs[0]["cumulative_sp_alu_pJ"] or 0)
    row["sfu_delta_pJ_vs_scale0"] = (row["cumulative_sfu_mul_pJ"] or 0) - (runs[0]["cumulative_sfu_mul_pJ"] or 0)
energies = [r["run_cumulative_gpu_J"] for r in runs]
monotonic = all(energies[i] <= energies[i + 1] + 1e-9 for i in range(len(energies) - 1))
record = {
    "image_index": int(image_index),
    "proxy": "uncalibrated; add/sub/convert use integer ALU weight 25; mul uses imul32 weight 4; div uses idiv weight 40; both mul and div feed the one McPAT MUL unit",
    "scales": [0, 0.5, 1, 2],
    "scale_meaning": "arbitrary multipliers on the stock integer weights, not calibrated error bars",
    "clock_mhz": 700.0,
    "sample_interval_cycles": 10000,
    "energy_scope": "Last GPU pJ is the lifetime sum of dynamic energy plus CONST_DYNAMIC. Leakage is absent. Per-kernel rows are not summed. The tail after the last 10000-cycle boundary is outside this energy.",
    "runs": runs,
    "monotonic_cumulative_energy": monotonic,
    "note": "Scale 0 is the no-compute-proxy control. Deltas are that run minus scale 0. A half scale is not exactly half after unsigned counter truncation. Not a float comparison and not a 2020-paper result.",
}
open(out, "w").write(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
if not monotonic:
    sys.exit(1)
PY
