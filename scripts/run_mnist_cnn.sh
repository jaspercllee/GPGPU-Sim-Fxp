#!/bin/sh
# Opt-in fresh MNIST CNN. Reuses data downloaded by scripts/run_mnist_mlp.sh.
# Does not run GPGPU-Sim.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
DATA="$ROOT/experiments/mnist_mlp/data"
EXP="$ROOT/experiments/mnist_cnn"
OUT="$EXP/out"
if [ ! -f "$DATA/train-images-idx3-ubyte.gz" ]; then
  echo "MNIST cache missing. Run scripts/run_mnist_mlp.sh once first." >&2
  exit 1
fi
mkdir -p "$OUT"
python3 "$EXP/train_mnist_cnn.py" "$DATA" "$OUT"
make -C "$EXP"
"$EXP/eval_fxp_cnn" "$DATA" "$OUT/weights.bin" "$OUT/eval.json"
python3 - << PY
import hashlib, json, os
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
out = "$OUT"
data = "$DATA"
mnist = []
base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
for name in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    mnist.append({"file": name, "sha256": sha(os.path.join(data, name)),
                  "source": base + name})
record = {
    "mnist": mnist,
    "train": json.load(open(out + "/train_meta.json")),
    "eval": json.load(open(out + "/eval.json")),
    "weights_sha256": sha(out + "/weights.bin"),
    "note": "New one-epoch CNN. Not the paper MLP and not a GPGPU-Sim run.",
}
text = json.dumps(record, indent=2) + "\n"
open(out + "/results.json", "w").write(text)
open("$EXP/results.json", "w").write(text)
print(text)
PY
