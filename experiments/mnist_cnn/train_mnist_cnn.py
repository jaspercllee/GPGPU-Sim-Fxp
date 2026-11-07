#!/usr/bin/env python3
"""One-epoch MNIST CNN. This is a new network, not the paper's 784-100-10 MLP.

conv 3x3 x4, valid, ReLU, 2x2 max-pool, then a fully connected layer to 10
logits. The paper does not specify this topology, optimizer, or loss.
"""
import gzip
import json
import os
import struct
import sys

import numpy as np

SEED = 20201112
LR = 0.1
BATCH = 64
N_TRAIN = 55000


def load_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051 or rows != 28 or cols != 28:
            raise SystemExit("bad image file %s" % path)
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    return raw.reshape(n, rows, cols).astype(np.float64) / 255.0


def load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise SystemExit("bad label file %s" % path)
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def conv_forward(x, w):
    n = x.shape[0]
    cols = np.empty((n, 26, 26, 9), dtype=np.float64)
    for ky in range(3):
        for kx in range(3):
            cols[:, :, :, ky * 3 + kx] = x[:, ky:ky + 26, kx:kx + 26]
    z = cols.reshape(-1, 9) @ w.reshape(4, 9).T
    return z.reshape(n, 26, 26, 4), cols


def conv_backward(cols, w, dz):
    flat = dz.reshape(-1, 4)
    dw = (flat.T @ cols.reshape(-1, 9)).reshape(4, 3, 3)
    dx_cols = (flat @ w.reshape(4, 9)).reshape(dz.shape[0], 26, 26, 9)
    dx = np.zeros((dz.shape[0], 28, 28), dtype=np.float64)
    for ky in range(3):
        for kx in range(3):
            dx[:, ky:ky + 26, kx:kx + 26] += dx_cols[:, :, :, ky * 3 + kx]
    return dx, dw


def pool_forward(h):
    a = h[:, 0::2, 0::2]
    b = h[:, 0::2, 1::2]
    c = h[:, 1::2, 0::2]
    d = h[:, 1::2, 1::2]
    m = np.maximum(np.maximum(a, b), np.maximum(c, d))
    mask = np.zeros(h.shape, dtype=np.float64)
    # Ties go to the first window entry. The mask is used only for training.
    eq_a = (m == a)
    eq_b = (m == b) & ~eq_a
    eq_c = (m == c) & ~eq_a & ~eq_b
    eq_d = ~(eq_a | eq_b | eq_c)
    mask[:, 0::2, 0::2] = eq_a
    mask[:, 0::2, 1::2] = eq_b
    mask[:, 1::2, 0::2] = eq_c
    mask[:, 1::2, 1::2] = eq_d
    return m, mask


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: train_mnist_cnn.py DATA_DIR OUT_DIR")
    data, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    images = load_idx_images(os.path.join(data, "train-images-idx3-ubyte.gz"))
    labels = load_idx_labels(os.path.join(data, "train-labels-idx1-ubyte.gz"))
    x_train, y_train = images[:N_TRAIN], labels[:N_TRAIN]
    rng = np.random.RandomState(SEED)
    wconv = rng.normal(0.0, np.sqrt(2.0 / 9.0), size=(4, 3, 3))
    bconv = np.zeros(4, dtype=np.float64)
    wfc = rng.normal(0.0, np.sqrt(2.0 / 676.0), size=(676, 10))
    bfc = np.zeros(10, dtype=np.float64)
    perm = rng.permutation(N_TRAIN)
    loss_sum = 0.0
    steps = 0
    for start in range(0, N_TRAIN, BATCH):
        idx = perm[start:start + BATCH]
        xb = x_train[idx]
        yb = y_train[idx]
        bsz = xb.shape[0]
        z, cols = conv_forward(xb, wconv)
        z = z + bconv.reshape(1, 1, 1, 4)
        h = np.maximum(z, 0.0)
        pooled, mask = pool_forward(h)
        flat = pooled.reshape(bsz, 676)
        logits = flat @ wfc + bfc
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        prob = exp / exp.sum(axis=1, keepdims=True)
        loss_sum += float(-np.log(prob[np.arange(bsz), yb] + 1e-12).mean())
        dlogits = prob.copy()
        dlogits[np.arange(bsz), yb] -= 1.0
        dlogits /= float(bsz)
        dwfc = flat.T @ dlogits
        dbfc = dlogits.sum(axis=0)
        dflat = dlogits @ wfc.T
        dpool = dflat.reshape(bsz, 13, 13, 4)
        dh = np.zeros_like(h)
        dh[:, 0::2, 0::2] = dpool * mask[:, 0::2, 0::2]
        dh[:, 0::2, 1::2] = dpool * mask[:, 0::2, 1::2]
        dh[:, 1::2, 0::2] = dpool * mask[:, 1::2, 0::2]
        dh[:, 1::2, 1::2] = dpool * mask[:, 1::2, 1::2]
        dz = dh * (z > 0.0)
        dx, dwconv = conv_backward(cols, wconv, dz)
        dbconv = dz.sum(axis=(0, 1, 2))
        wconv -= LR * dwconv
        bconv -= LR * dbconv
        wfc -= LR * dwfc
        bfc -= LR * dbfc
        steps += 1
        del dx
    path = os.path.join(out, "weights.bin")
    with open(path, "wb") as f:
        f.write(np.asarray([4, 3, 676, 10], dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(wconv, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(bconv, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(wfc, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(bfc, dtype=np.float64).tobytes())
    meta = {
        "network": "new MNIST CNN, not the paper MLP",
        "topology": "conv 3x3x4 valid, ReLU, max-pool 2x2, FC 676x10 logits",
        "seed": SEED,
        "optimizer": "sgd",
        "learning_rate": LR,
        "batch_size": BATCH,
        "epochs": 1,
        "loss": "softmax cross-entropy on float logits",
        "init": "He normal",
        "input": "pixel/255",
        "n_train_sgd": N_TRAIN,
        "n_calibration_held_out": 5000,
        "steps": steps,
        "mean_batch_cross_entropy": loss_sum / steps,
    }
    with open(os.path.join(out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print("trained steps", steps, "mean ce", meta["mean_batch_cross_entropy"])


if __name__ == "__main__":
    main()
