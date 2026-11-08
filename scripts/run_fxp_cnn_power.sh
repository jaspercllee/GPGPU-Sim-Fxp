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
sudo cp "$ROOT/src/gpgpu-sim/shader.cc" /var/tmp/gpgpu-src/src/gpgpu-sim/
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
sudo rm -f /var/tmp/gpgpu-src/build/gcc-4.8.4/cuda-7050/release/gpgpu-sim/gpu-sim.o
sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /src; source setup_environment; make -j2'
sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /tmp/run; nvcc -cudart shared -arch=sm_20 fxp_cnn_host.cu -o fxp_cnn_host'
sudo rm -f "$CHROOT/tmp/run"/gpgpusim_power_report__*.log
START=$(date +%s)
set +e
sudo chroot "$CHROOT" bash -lc 'export CUDA_INSTALL_PATH=/usr/local/cuda-7.5; export PATH=$CUDA_INSTALL_PATH/bin:$PATH; cd /tmp/run; PTX_SIM_USE_PTX_FILE=1 PTX_SIM_KERNELFILE=/tmp/run/fxp_cnn_image.ptx LD_LIBRARY_PATH=/src/lib/gcc-4.8.4/cuda-7050/release ./fxp_cnn_host /tmp/run/sim_case.bin' > "$OUT/sim_cnn_power.log" 2>&1
STATUS=$?
set -e
END=$(date +%s)
echo "HOST_EXIT $STATUS WALL_SEC $((END - START))" | tee -a "$OUT/sim_cnn_power.log"
python3 - "$OUT/sim_cnn_power.log" "$STATUS" "$((END - START))" "$INDEX" "$CHROOT/tmp/run" "$ROOT/experiments/legacy_ptx/cnn_one_image_power.json" << 'PY'
import glob, json, os, re, sys
log, status, wall, index, rundir, out = sys.argv[1:]
text = open(log, errors="replace").read()
reports = sorted(glob.glob(os.path.join(rundir, "gpgpusim_power_report__*.log")))
report_text = open(reports[-1], errors="replace").read() if reports else ""
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
# The GPU pJ column is the lifetime GPUenergy accumulator, reprinted at each dump.
named = []
for name, block in blocks(report_text):
    named.append({
        "name": name,
        "cumulative_gpu_pJ": grab(block, r"GPU\|[^|\n]*\|[ \t]*([0-9]+(?:\.[0-9]+)?)"),
        "printed_avg_power_W_rounded": grab(block, r"kernel_avg_power = ([0-9]+(?:\.[0-9]+)?)"),
        "samples_in_kernel_total_inst": grab(block, r"Total Inst\|[ \t]*([0-9]+(?:\.[0-9]+)?)"),
        "fxp_int_per_sample_printed": grab(block, r"gpu_avg_FXP_INT, = ([0-9]+(?:\.[0-9]+)?)"),
        "fp_int_per_sample_printed": grab(block, r"gpu_avg_FP_INT, = ([0-9]+(?:\.[0-9]+)?)"),
    })
kernel_cycles = []
lines = text.splitlines()
for i, ln in enumerate(lines):
    if not ln.startswith("kernel_name ="):
        continue
    name = ln.split("=", 1)[1].strip()
    cyc = None
    for nxt in lines[i + 1:i + 8]:
        m = re.search(r"gpu_sim_cycle = (\d+)", nxt)
        if m:
            cyc = int(m.group(1))
            break
    if name and cyc and (not kernel_cycles or kernel_cycles[-1]["name"] != name
                         or kernel_cycles[-1]["gpu_sim_cycle"] != cyc):
        kernel_cycles.append({"name": name, "gpu_sim_cycle": cyc,
                              "modeled_time_sec": cyc / 700e6})
def field(name):
    m = re.search(r"SIMCNN %s ([^\n]*)" % name, text)
    return m.group(1).strip() if m else ""
def ints(s):
    return [int(x) for x in s.split()] if s else []
cycles = [int(x) for x in re.findall(r"gpu_tot_sim_cycle = (\d+)", text)]
record = {
    "producer": "bespoke loop PTX, timing mode, GPUWattch enabled",
    "image_index": int(index),
    "host_exit": int(status),
    "host_wall_sec": int(wall),
    "functional_mode": "mode=functional simulation" in text,
    "power_line": next((ln.strip() for ln in text.splitlines() if ln.strip().startswith("-power_simulation_enabled") or "power_simulation_enabled" in ln and "Turn on" in ln), ""),
    "sample_stat": next((ln.strip() for ln in text.splitlines() if "gpgpu_runtime_stat" in ln), ""),
    "fxp_assumption": next((ln.strip() for ln in text.splitlines() if ln.startswith("GPUWattch FXP:")), ""),
    "clock_mhz": 700.0,
    "clock_note": "gpgpusim.config core clock and gpuwattch XML clock_rate are both 700 MHz",
    "last_gpu_tot_sim_cycle": cycles[-1] if cycles else None,
    "modeled_time_sec": (cycles[-1] / 700e6) if cycles else None,
    "sample_interval_cycles": 10000,
    "sample_window_sec": 10000 / 700e6,
    "kernel_cycles": kernel_cycles,
    "cumulative_gpu_prints": named,
    "run_cumulative_gpu_pJ": named[-1]["cumulative_gpu_pJ"] if named else None,
    "run_cumulative_gpu_J": (named[-1]["cumulative_gpu_pJ"] * 1e-12) if named and named[-1]["cumulative_gpu_pJ"] is not None else None,
    "power_report_files": [os.path.basename(p) for p in reports],
    "energy_scope": "GPU pJ is lifetime dynamic plus CONST_DYNAMIC energy, not per kernel and not leakage. kernel_avg_power is printed at 0 decimal places.",
    "do_not_sum_kernel_rows": True,
    "sim_logits": ints(field("sim_logits")),
    "cpu_fxp_logits": ints(field("cpu_fxp_logits")),
    "logit_mismatch": field("logit_mismatch"),
    "conv_mismatch": field("conv_mismatch"),
    "summary": field("sim_argmax"),
    "simcnn_status": field("status"),
    "toolchain": "Ubuntu 14.04 chroot, gcc 4.8.4, CUDA 7.5, GPGPU-Sim 3.2.2, GTX480 XML",
    "note": "The GPU pJ column is a running total of dynamic energy plus CONST_DYNAMIC. It is not leakage and not a per-kernel energy. Fixed-point calls are absent from the SP/SFU/FPU units; the count reaches the instruction window and the dependency checker only. Not a calibrated FXP ALU measurement, not a float comparison, and not a 10000-image result.",
}
open(out, "w").write(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
sys.exit(0 if int(status) == 0 else 1)
PY
