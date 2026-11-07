// Float baseline and fixed-point CPU inference for a fresh MNIST MLP.
// MACs call spirv::binary_mul and binary_add. Sigmoid stays in double.
// This program does not run GPGPU-Sim.

#include "fxp_requant.h"
#include "../../src/cuda-sim/spirv_fxp_arith.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kIn = 784;
constexpr int kHid = 100;
constexpr int kOut = 10;

struct Model {
  std::vector<double> w1, b1, w2, b2;
};

struct FxpStats {
  unsigned long long product_overflows = 0;
  unsigned long long convert_clips = 0;
  unsigned long long macs = 0;
};

int argmax10(const double* y) {
  int best = 0;
  for (int i = 1; i < kOut; ++i)
    if (y[i] > y[best]) best = i;
  return best;
}

double sigmoid(double z) {
  if (z > 40.0) return 1.0;
  if (z < -40.0) return 0.0;
  return 1.0 / (1.0 + std::exp(-z));
}

void forward_float(const Model& m, const double* x, double* y) {
  double h[kHid];
  for (int j = 0; j < kHid; ++j) {
    double z = m.b1[j];
    for (int i = 0; i < kIn; ++i)
      z += x[i] * m.w1[i * kHid + j];
    h[j] = sigmoid(z);
  }
  for (int j = 0; j < kOut; ++j) {
    double z = m.b2[j];
    for (int i = 0; i < kHid; ++i)
      z += h[i] * m.w2[i * kOut + j];
    y[j] = sigmoid(z);
  }
}

int32_t quantize_counted(double x, int exp, int width, FxpStats& st) {
  int scale_exp = -exp;
  double mag = x < 0 ? -x : x;
  double scaled = mag;
  if (scale_exp >= 0 && scale_exp < 62)
    scaled *= double(int64_t(1) << scale_exp);
  else if (scale_exp < 0 && -scale_exp < 62)
    scaled /= double(int64_t(1) << -scale_exp);
  else
    scaled = 0;
  int64_t q = int64_t(scaled + 0.5);
  if (x < 0) q = -q;
  int32_t sat = int32_t(fxp_sat_signed(q, width));
  if (int64_t(sat) != q) st.convert_clips++;
  return sat;
}

void mac_layer(const int32_t* x, int x_n, const int32_t* w, const double* bias,
               int y_n, int in_exp, int w_exp, int acc_exp,
               int in_bits, int w_bits, std::vector<double>& y, FxpStats& st) {
  y.assign(y_n, 0.0);
  for (int j = 0; j < y_n; ++j) {
    int32_t acc = quantize_counted(bias[j], acc_exp, 32, st);
    for (int i = 0; i < x_n; ++i) {
      const int32_t q_w = w[i * y_n + j];
      const int64_t wide = int64_t(x[i]) * int64_t(q_w);
      if (in_bits == 32 && w_bits == 32 &&
          (wide > int64_t(INT32_MAX) || wide < int64_t(INT32_MIN)))
        st.product_overflows++;

      spirv::_detail::fxp::FxpBinaryFuncOpTy a = {0};
      spirv::_detail::fxp::FxpBinaryFuncOpTy b = {0};
      spirv::_detail::fxp::FxpBinaryFuncOpTy prod = {0};
      if (in_bits <= 8) a.Single = uint8_t(uint32_t(x[i]));
      else a.Full = uint32_t(x[i]);
      if (w_bits <= 8) b.Single = uint8_t(uint32_t(q_w));
      else b.Full = uint32_t(q_w);

      const unsigned in_size = unsigned(in_bits / 8);
      const unsigned w_size = unsigned(w_bits / 8);
      unsigned prod_size = in_size + w_size;
      if (!(in_size == w_size && (in_size + w_size) * 8u <= 32u))
        prod_size = 4;
      int prod_exp = 0;
      spirv::_detail::fxp::binary_mul(in_size, a, in_exp, w_size, b, w_exp,
                                     prod_size, prod, prod_exp);
      const int32_t raw_prod = (prod_size == 2)
        ? int32_t(int16_t(prod.Half))
        : int32_t(prod.Full);
      const int32_t aligned = int32_t(fxp_requantize(raw_prod, prod_exp, acc_exp, 32));

      spirv::_detail::fxp::FxpBinaryFuncOpTy acc_op = {0};
      spirv::_detail::fxp::FxpBinaryFuncOpTy add_op = {0};
      spirv::_detail::fxp::FxpBinaryFuncOpTy sum = {0};
      acc_op.Full = uint32_t(acc);
      add_op.Full = uint32_t(aligned);
      int sum_exp = 0;
      spirv::_detail::fxp::binary_add(4, acc_op, acc_exp, 4, add_op, acc_exp,
                                     4, sum, sum_exp);
      acc = int32_t(sum.Full);
      st.macs++;
    }
    y[j] = sigmoid(double(acc) * std::ldexp(1.0, acc_exp));
  }
}

void eval_fxp(const Model& m, const std::vector<double>& images,
              int io_exp, int io_bits, int acc_exp,
              std::vector<int>& pred, std::vector<double>& outputs, FxpStats& st) {
  const int n = int(images.size() / kIn);
  pred.resize(n);
  outputs.assign(n * kOut, 0.0);
  std::vector<int32_t> qw1(kIn * kHid), qw2(kHid * kOut);
  for (int i = 0; i < kIn * kHid; ++i)
    qw1[i] = quantize_counted(m.w1[i], io_exp, io_bits, st);
  for (int i = 0; i < kHid * kOut; ++i)
    qw2[i] = quantize_counted(m.w2[i], io_exp, io_bits, st);

  std::vector<int32_t> xq(kIn), hq(kHid);
  std::vector<double> h, y;
  for (int n_i = 0; n_i < n; ++n_i) {
    const double* x = &images[n_i * kIn];
    for (int i = 0; i < kIn; ++i)
      xq[i] = quantize_counted(x[i], io_exp, io_bits, st);
    mac_layer(xq.data(), kIn, qw1.data(), m.b1.data(), kHid,
              io_exp, io_exp, acc_exp, io_bits, io_bits, h, st);
    for (int i = 0; i < kHid; ++i)
      hq[i] = quantize_counted(h[i], io_exp, io_bits, st);
    mac_layer(hq.data(), kHid, qw2.data(), m.b2.data(), kOut,
              io_exp, io_exp, acc_exp, io_bits, io_bits, y, st);
    for (int j = 0; j < kOut; ++j)
      outputs[n_i * kOut + j] = y[j];
    pred[n_i] = argmax10(y.data());
    if ((n_i + 1) % 2000 == 0)
      std::fprintf(stderr, "fxp exp %d images %d/%d\n", io_exp, n_i + 1, n);
  }
}

bool load_images(const std::string& path, std::vector<double>& out, int expect_n) {
  std::ifstream in(path, std::ios::binary);
  uint32_t magic = 0, n = 0, rows = 0, cols = 0;
  in.read(reinterpret_cast<char*>(&magic), 4);
  in.read(reinterpret_cast<char*>(&n), 4);
  in.read(reinterpret_cast<char*>(&rows), 4);
  in.read(reinterpret_cast<char*>(&cols), 4);
  magic = __builtin_bswap32(magic);
  n = __builtin_bswap32(n);
  rows = __builtin_bswap32(rows);
  cols = __builtin_bswap32(cols);
  if (!in || magic != 2051 || rows != 28 || cols != 28 || int(n) != expect_n)
    return false;
  std::vector<uint8_t> raw(size_t(n) * 784);
  in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
  out.resize(raw.size());
  for (size_t i = 0; i < raw.size(); ++i)
    out[i] = double(raw[i]) / 255.0;
  return bool(in);
}

bool load_labels(const std::string& path, std::vector<int>& out, int expect_n) {
  std::ifstream in(path, std::ios::binary);
  uint32_t magic = 0, n = 0;
  in.read(reinterpret_cast<char*>(&magic), 4);
  in.read(reinterpret_cast<char*>(&n), 4);
  magic = __builtin_bswap32(magic);
  n = __builtin_bswap32(n);
  if (!in || magic != 2049 || int(n) != expect_n)
    return false;
  std::vector<uint8_t> raw(n);
  in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
  out.assign(raw.begin(), raw.end());
  return bool(in);
}

bool load_weights(const std::string& path, Model& m) {
  std::ifstream in(path, std::ios::binary);
  int32_t dims[3] = {0, 0, 0};
  in.read(reinterpret_cast<char*>(dims), sizeof(dims));
  if (!in || dims[0] != kIn || dims[1] != kHid || dims[2] != kOut)
    return false;
  m.w1.resize(kIn * kHid);
  m.b1.resize(kHid);
  m.w2.resize(kHid * kOut);
  m.b2.resize(kOut);
  in.read(reinterpret_cast<char*>(m.w1.data()), std::streamsize(m.w1.size() * sizeof(double)));
  in.read(reinterpret_cast<char*>(m.b1.data()), std::streamsize(m.b1.size() * sizeof(double)));
  in.read(reinterpret_cast<char*>(m.w2.data()), std::streamsize(m.w2.size() * sizeof(double)));
  in.read(reinterpret_cast<char*>(m.b2.data()), std::streamsize(m.b2.size() * sizeof(double)));
  return bool(in);
}

int count_correct(const std::vector<int>& pred, const std::vector<int>& labels) {
  int ok = 0;
  for (size_t i = 0; i < pred.size(); ++i)
    if (pred[i] == labels[i]) ok++;
  return ok;
}

double mean_abs(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i)
    s += std::fabs(a[i] - b[i]);
  return s / double(a.size());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: eval_fxp_mlp DATA_DIR WEIGHTS OUT_JSON\n");
    return 2;
  }
  Model model;
  if (!load_weights(argv[2], model)) {
    std::fprintf(stderr, "bad weights\n");
    return 1;
  }
  const std::string data = argv[1];
  std::vector<double> test_x;
  std::vector<int> test_y;
  if (!load_images(data + "/t10k-images-idx3-ubyte", test_x, 10000) ||
      !load_labels(data + "/t10k-labels-idx1-ubyte", test_y, 10000)) {
    std::fprintf(stderr, "bad mnist idx\n");
    return 1;
  }

  std::vector<double> float_y(10000 * kOut);
  std::vector<int> float_pred(10000);
  for (int i = 0; i < 10000; ++i) {
    double y[kOut];
    forward_float(model, &test_x[i * kIn], y);
    for (int j = 0; j < kOut; ++j) float_y[i * kOut + j] = y[j];
    float_pred[i] = argmax10(y);
  }

  FxpStats st32, st8;
  std::vector<int> pred32, pred8;
  std::vector<double> y32, y8;
  // 32-bit storage at exponent -12. The mul helper returns exponent -24 in
  // 32 bits, then requantize shifts that raw value to accumulator exponent -12.
  eval_fxp(model, test_x, -12, 32, -12, pred32, y32, st32);
  // 8-bit inputs/weights at exponent -4. Mul of equal 8-bit sizes returns 16
  // bits at exponent -8. Accumulator is 32-bit at that same exponent -8.
  eval_fxp(model, test_x, -4, 8, -8, pred8, y8, st8);

  const int float_ok = count_correct(float_pred, test_y);
  const int fxp32_ok = count_correct(pred32, test_y);
  const int fxp8_ok = count_correct(pred8, test_y);
  FILE* out = std::fopen(argv[3], "w");
  if (!out) return 1;
  std::fprintf(out,
    "{\n"
    "  \"test_images\": 10000,\n"
    "  \"float_correct\": %d,\n"
    "  \"fxp32_exp_m12_correct\": %d,\n"
    "  \"fxp8_exp_m4_correct\": %d,\n"
    "  \"mae_fxp32_vs_float_sigmoid\": %.8g,\n"
    "  \"mae_fxp8_vs_float_sigmoid\": %.8g,\n"
    "  \"fxp32_product_overflows\": %llu,\n"
    "  \"fxp8_product_overflows\": %llu,\n"
    "  \"fxp32_convert_clips\": %llu,\n"
    "  \"fxp8_convert_clips\": %llu,\n"
    "  \"fxp32_macs\": %llu,\n"
    "  \"sigmoid\": \"float64 between layers and on the logits\",\n"
    "  \"simulator\": false\n"
    "}\n",
    float_ok, fxp32_ok, fxp8_ok,
    mean_abs(y32, float_y), mean_abs(y8, float_y),
    st32.product_overflows, st8.product_overflows,
    st32.convert_clips, st8.convert_clips, st32.macs);
  std::fclose(out);
  std::printf("float %d/10000 fxp32 %d/10000 fxp8 %d/10000\n",
              float_ok, fxp32_ok, fxp8_ok);
  std::printf("fxp32 overflows %llu clips %llu fxp8 overflows %llu clips %llu\n",
              st32.product_overflows, st32.convert_clips,
              st8.product_overflows, st8.convert_clips);
  return 0;
}
