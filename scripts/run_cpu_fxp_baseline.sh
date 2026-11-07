#!/bin/sh
# Re-run the CPU fixed-point baseline and print the version strings that are
# actually in this tree. This script does not launch GPGPU-Sim.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"

echo "version file:"
cat "$ROOT/version"
echo "CHANGES heading:"
sed -n '1,4p' "$ROOT/CHANGES"
echo "committed config directories:"
find "$ROOT/configs" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
echo "PTX/CUDA/OpenCL workloads in tree:"
find "$ROOT" -type f \( -name '*.ptx' -o -name '*.cu' -o -name '*.cl' \) ! -path '*/third_party/*' ! -path '*/build/*' | sed "s#^$ROOT/##" || true

make -C "$ROOT/tests/cpu_fxp"
echo "cpu fixed-point baseline: ok"
echo "GPU timing, CUDA runtime, and OpenCL paths were not executed."
