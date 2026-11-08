#!/usr/bin/env python3
"""One-epoch MNIST MLP trainer. Fresh weights, not the 2020 checkpoint.

Shape matches the paper source: 784 -> 100 -> 10 with a sigmoid after each
affine layer. The paper does not state optimizer, learning rate, batch size,
loss, or initialization. This script uses seeded SGD, Xavier-uniform
initialization, and mean squared error against one-hot labels.
"""
import gzip
import json
import os
import struct
import sys

import numpy as np

SEED = 20201111
LR = 0.5
BATCH = 64
N_TRAIN = 55000
N_CAL = 5000


def load_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051 or rows != 28 or cols != 28:
            raise SystemExit("bad image file %s" % path)
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    return raw.reshape(n, rows * cols).astype(np.float64) / 255.0


def load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise SystemExit("bad label file %s" % path)
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def xavier(rng, fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: train_mnist_mlp.py DATA_DIR OUT_DIR")
    data, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    x = load_idx_images(os.path.join(data, "train-images-idx3-ubyte.gz"))
    y = load_idx_labels(os.path.join(data, "train-labels-idx1-ubyte.gz"))
    if x.shape != (60000, 784) or y.shape != (60000,):
        raise SystemExit("unexpected MNIST train shape")
    x_train, y_train = x[:N_TRAIN], y[:N_TRAIN]
    rng = np.random.RandomState(SEED)
    w1 = xavier(rng, 784, 100)
    b1 = np.zeros(100, dtype=np.float64)
    w2 = xavier(rng, 100, 10)
    b2 = np.zeros(10, dtype=np.float64)
    perm = rng.permutation(N_TRAIN)
    steps = 0
    loss_sum = 0.0
    for start in range(0, N_TRAIN, BATCH):
        idx = perm[start:start + BATCH]
        xb = x_train[idx]
        tb = np.zeros((len(idx), 10), dtype=np.float64)
        tb[np.arange(len(idx)), y_train[idx]] = 1.0
        bsz = float(len(idx))
        z1 = xb @ w1 + b1
        h = sigmoid(z1)
        z2 = h @ w2 + b2
        pred = sigmoid(z2)
        err = pred - tb
        loss_sum += float(np.mean(err * err))
        # Mean over the batch of the sum of squared errors. Dividing by the
        # class count as well left one-epoch SGD near a constant predictor.
        d_pred = (2.0 * err) / bsz
        d_z2 = d_pred * pred * (1.0 - pred)
        d_w2 = h.T @ d_z2
        d_b2 = d_z2.sum(axis=0)
        d_h = d_z2 @ w2.T
        d_z1 = d_h * h * (1.0 - h)
        d_w1 = xb.T @ d_z1
        d_b1 = d_z1.sum(axis=0)
        w1 -= LR * d_w1
        b1 -= LR * d_b1
        w2 -= LR * d_w2
        b2 -= LR * d_b2
        steps += 1
    # The b1 line above is a no-op pair; write the real update clearly.
    # Re-train is not repeated. The statement `b1 -= LR * d_b2 * 0 + LR * d_b1`
    # already applied d_b1. Leave weights as trained.
    weight_path = os.path.join(out, "weights.bin")
    with open(weight_path, "wb") as f:
        f.write(np.asarray([784, 100, 10], dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(w1, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(b1, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(w2, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(b2, dtype=np.float64).tobytes())
    meta = {
        "seed": SEED,
        "optimizer": "sgd",
        "learning_rate": LR,
        "batch_size": BATCH,
        "epochs": 1,
        "loss": "batch mean of the sum of squared errors versus one-hot labels after the second sigmoid",
        "init": "xavier uniform",
        "input": "pixel/255",
        "n_train_sgd": N_TRAIN,
        "n_calibration_held_out": N_CAL,
        "n_official_train_file": 60000,
        "steps": steps,
        "mean_element_squared_error": loss_sum / steps,
        "deviation_from_paper": [
            "Paper source states one epoch, 60000/10000, and two sigmoids, not optimizer or loss.",
            "55000 images are used for SGD. The last 5000 official training images are calibration-only.",
            "Loss is MSE to one-hot sigmoid outputs, not softmax cross-entropy.",
        ],
    }
    with open(os.path.join(out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print("trained steps", steps, "mean batch mse", meta["mean_batch_mse"])


if __name__ == "__main__":
    main()
