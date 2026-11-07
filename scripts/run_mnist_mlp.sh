#!/bin/sh
# Opt-in fresh MNIST MLP. Downloads the public dataset, trains one epoch, and
# evaluates float plus fixed-point CPU inference. Does not run GPGPU-Sim.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
EXP="$ROOT/experiments/mnist_mlp"
DATA="$EXP/data"
OUT="$EXP/out"
mkdir -p "$DATA" "$OUT"

BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  if [ ! -f "$DATA/$f" ]; then
    curl -fsSL -o "$DATA/$f" "$BASE/$f"
  fi
  gunzip -c "$DATA/$f" > "$DATA/${f%.gz}"
done

python3 - << PY
import hashlib, json, os
data = "$DATA"
files = [
  "train-images-idx3-ubyte.gz",
  "train-labels-idx1-ubyte.gz",
  "t10k-images-idx3-ubyte.gz",
  "t10k-labels-idx1-ubyte.gz",
]
rows = []
for name in files:
    h = hashlib.sha256()
    with open(os.path.join(data, name), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    rows.append({"file": name, "sha256": h.hexdigest(),
                 "source": "$BASE/" + name})
with open("$OUT/mnist_sha256.json", "w") as f:
    json.dump(rows, f, indent=2)
    f.write("\n")
for row in rows:
    print(row["sha256"], row["file"])
PY

python3 "$EXP/train_mnist_mlp.py" "$DATA" "$OUT"
make -C "$EXP"
"$EXP/eval_fxp_mlp" "$DATA" "$OUT/weights.bin" "$OUT/eval.json"
python3 - << PY
import hashlib, json
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
out = "$OUT"
record = {
    "mnist": json.load(open(out + "/mnist_sha256.json")),
    "train": json.load(open(out + "/train_meta.json")),
    "eval": json.load(open(out + "/eval.json")),
    "weights_sha256": sha(out + "/weights.bin"),
    "note": "Fresh one-epoch weights. Not the 2020 paper checkpoint. CPU only; GPGPU-Sim was not run.",
}
text = json.dumps(record, indent=2) + "\n"
open(out + "/results.json", "w").write(text)
open("$EXP/results.json", "w").write(text)
print(text)
PY
