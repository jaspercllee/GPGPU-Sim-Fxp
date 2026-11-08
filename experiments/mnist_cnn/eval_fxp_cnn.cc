// Float and fixed-point CPU inference for the fresh MNIST CNN.
// Conv and FC MACs call binary_mul and binary_add. ReLU is a signed compare
// on the accumulator raw value. Max-pool is an integer max at that same
// exponent. Neither is a new PTX opcode. Sigmoid is not used.
// This program does not run GPGPU-Sim.

#include "../mnist_mlp/fxp_requant.h"
#include "../../src/cuda-sim/spirv_fxp_arith.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kOc = 4;
constexpr int kFlat = 13 * 13 * 4;

struct Model {
  std::vector<double> wconv, bconv, wfc, bfc;
};

struct Stats {
  unsigned long long product_overflows = 0;
  unsigned long long convert_clips = 0;
  unsigned long long macs = 0;
  unsigned long long relu_zeros = 0;
};

int32_t quantize(double x, int exp, int width, Stats& st) {
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

int32_t mac(int32_t qa, int qa_exp, int qa_bits, int32_t qb, int qb_exp,
            int qb_bits, int32_t acc, int acc_exp, Stats& st) {
  const int64_t wide = int64_t(qa) * int64_t(qb);
  if (qa_bits == 32 && qb_bits == 32 &&
      (wide > int64_t(INT32_MAX) || wide < int64_t(INT32_MIN)))
    st.product_overflows++;
  spirv::_detail::fxp::FxpBinaryFuncOpTy a = {0}, b = {0}, prod = {0};
  if (qa_bits <= 8) a.Single = uint8_t(uint32_t(qa));
  else a.Full = uint32_t(qa);
  if (qb_bits <= 8) b.Single = uint8_t(uint32_t(qb));
  else b.Full = uint32_t(qb);
  const unsigned asz = unsigned(qa_bits / 8);
  const unsigned bsz = unsigned(qb_bits / 8);
  unsigned psz = asz + bsz;
  if (!(asz == bsz && (asz + bsz) * 8u <= 32u)) psz = 4;
  int prod_exp = 0;
  spirv::_detail::fxp::binary_mul(asz, a, qa_exp, bsz, b, qb_exp, psz, prod, prod_exp);
  const int32_t raw = (psz == 2) ? int32_t(int16_t(prod.Half)) : int32_t(prod.Full);
  const int32_t aligned = int32_t(fxp_requantize(raw, prod_exp, acc_exp, 32));
  spirv::_detail::fxp::FxpBinaryFuncOpTy acc_op = {0}, add_op = {0}, sum = {0};
  acc_op.Full = uint32_t(acc);
  add_op.Full = uint32_t(aligned);
  int sum_exp = 0;
  spirv::_detail::fxp::binary_add(4, acc_op, acc_exp, 4, add_op, acc_exp, 4, sum, sum_exp);
  st.macs++;
  return int32_t(sum.Full);
}

void forward_float(const Model& m, const double* image, double* logits) {
  double conv[26 * 26 * 4];
  for (int oc = 0; oc < 4; ++oc) {
    for (int y = 0; y < 26; ++y) {
      for (int x = 0; x < 26; ++x) {
        double z = m.bconv[oc];
        for (int ky = 0; ky < 3; ++ky)
          for (int kx = 0; kx < 3; ++kx)
            z += image[(y + ky) * 28 + (x + kx)] *
                 m.wconv[oc * 9 + ky * 3 + kx];
        conv[(y * 26 + x) * 4 + oc] = z > 0.0 ? z : 0.0;
      }
    }
  }
  double pooled[kFlat];
  for (int y = 0; y < 13; ++y) {
    for (int x = 0; x < 13; ++x) {
      for (int oc = 0; oc < 4; ++oc) {
        double best = conv[((y * 2) * 26 + (x * 2)) * 4 + oc];
        best = std::max(best, conv[((y * 2) * 26 + (x * 2 + 1)) * 4 + oc]);
        best = std::max(best, conv[((y * 2 + 1) * 26 + (x * 2)) * 4 + oc]);
        best = std::max(best, conv[((y * 2 + 1) * 26 + (x * 2 + 1)) * 4 + oc]);
        pooled[(y * 13 + x) * 4 + oc] = best;
      }
    }
  }
  for (int j = 0; j < 10; ++j) {
    double z = m.bfc[j];
    for (int i = 0; i < kFlat; ++i)
      z += pooled[i] * m.wfc[i * 10 + j];
    logits[j] = z;
  }
}

void forward_fxp(const Model& m, const double* image, double* logits,
                 int io_exp, int io_bits, int acc_exp, Stats& st) {
  std::vector<int32_t> qw(36), qfc(kFlat * 10), qimg(28 * 28);
  for (int i = 0; i < 36; ++i)
    qw[i] = quantize(m.wconv[i], io_exp, io_bits, st);
  for (int i = 0; i < kFlat * 10; ++i)
    qfc[i] = quantize(m.wfc[i], io_exp, io_bits, st);
  for (int i = 0; i < 28 * 28; ++i)
    qimg[i] = quantize(image[i], io_exp, io_bits, st);
  std::vector<int32_t> conv(26 * 26 * 4);
  for (int oc = 0; oc < 4; ++oc) {
    int32_t qbias = quantize(m.bconv[oc], acc_exp, 32, st);
    for (int y = 0; y < 26; ++y) {
      for (int x = 0; x < 26; ++x) {
        int32_t acc = qbias;
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            int32_t pix = qimg[(y + ky) * 28 + (x + kx)];
            int32_t wt = qw[oc * 9 + ky * 3 + kx];
            acc = mac(pix, io_exp, io_bits, wt, io_exp, io_bits, acc, acc_exp, st);
          }
        }
        if (acc < 0) {
          acc = 0;
          st.relu_zeros++;
        }
        conv[(y * 26 + x) * 4 + oc] = acc;
      }
    }
  }
  std::vector<int32_t> pooled(kFlat);
  for (int y = 0; y < 13; ++y) {
    for (int x = 0; x < 13; ++x) {
      for (int oc = 0; oc < 4; ++oc) {
        int32_t best = conv[((y * 2) * 26 + (x * 2)) * 4 + oc];
        best = std::max(best, conv[((y * 2) * 26 + (x * 2 + 1)) * 4 + oc]);
        best = std::max(best, conv[((y * 2 + 1) * 26 + (x * 2)) * 4 + oc]);
        best = std::max(best, conv[((y * 2 + 1) * 26 + (x * 2 + 1)) * 4 + oc]);
        pooled[(y * 13 + x) * 4 + oc] = best;
      }
    }
  }
  for (int j = 0; j < 10; ++j) {
    int32_t acc = quantize(m.bfc[j], acc_exp, 32, st);
    for (int i = 0; i < kFlat; ++i) {
      int32_t act = pooled[i];
      int act_bits = 32;
      int act_exp = acc_exp;
      if (io_bits == 8) {
        act = quantize(double(pooled[i]) * std::ldexp(1.0, acc_exp), io_exp, 8, st);
        act_bits = 8;
        act_exp = io_exp;
      }
      acc = mac(act, act_exp, act_bits, qfc[i * 10 + j], io_exp, io_bits,
                acc, acc_exp, st);
    }
    logits[j] = double(acc) * std::ldexp(1.0, acc_exp);
  }
}

int argmax10(const double* y) {
  int best = 0;
  for (int i = 1; i < 10; ++i)
    if (y[i] > y[best]) best = i;
  return best;
}

bool load_images(const std::string& path, std::vector<double>& out) {
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
  if (!in || magic != 2051 || n != 10000 || rows != 28 || cols != 28)
    return false;
  std::vector<uint8_t> raw(10000 * 784);
  in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
  out.resize(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) out[i] = double(raw[i]) / 255.0;
  return bool(in);
}

bool load_labels(const std::string& path, std::vector<int>& out) {
  std::ifstream in(path, std::ios::binary);
  uint32_t magic = 0, n = 0;
  in.read(reinterpret_cast<char*>(&magic), 4);
  in.read(reinterpret_cast<char*>(&n), 4);
  magic = __builtin_bswap32(magic);
  n = __builtin_bswap32(n);
  if (!in || magic != 2049 || n != 10000) return false;
  std::vector<uint8_t> raw(n);
  in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
  out.assign(raw.begin(), raw.end());
  return bool(in);
}

bool load_weights(const std::string& path, Model& m) {
  std::ifstream in(path, std::ios::binary);
  int32_t dims[4] = {0, 0, 0, 0};
  in.read(reinterpret_cast<char*>(dims), sizeof(dims));
  if (!in || dims[0] != 4 || dims[1] != 3 || dims[2] != 676 || dims[3] != 10)
    return false;
  m.wconv.resize(36);
  m.bconv.resize(4);
  m.wfc.resize(kFlat * 10);
  m.bfc.resize(10);
  in.read(reinterpret_cast<char*>(m.wconv.data()), std::streamsize(36 * sizeof(double)));
  in.read(reinterpret_cast<char*>(m.bconv.data()), std::streamsize(4 * sizeof(double)));
  in.read(reinterpret_cast<char*>(m.wfc.data()), std::streamsize(m.wfc.size() * sizeof(double)));
  in.read(reinterpret_cast<char*>(m.bfc.data()), std::streamsize(10 * sizeof(double)));
  return bool(in);
}

struct Layers32 {
  int32_t image[784];
  int32_t wconv[36];
  int32_t bconv[4];
  int32_t conv[26 * 26 * 4];
  int32_t pool[kFlat];
  int32_t wfc[kFlat * 10];
  int32_t bfc[10];
  int32_t logits[10];
};

// Same 32-bit contract as forward_fxp(..., -12, 32, -12). Logits stay raw.
void forward_fxp32_layers(const Model& m, const double* image, Layers32& L, Stats& st) {
  for (int i = 0; i < 36; ++i)
    L.wconv[i] = quantize(m.wconv[i], -12, 32, st);
  for (int i = 0; i < kFlat * 10; ++i)
    L.wfc[i] = quantize(m.wfc[i], -12, 32, st);
  for (int i = 0; i < 784; ++i)
    L.image[i] = quantize(image[i], -12, 32, st);
  for (int oc = 0; oc < 4; ++oc) {
    L.bconv[oc] = quantize(m.bconv[oc], -12, 32, st);
    for (int y = 0; y < 26; ++y) {
      for (int x = 0; x < 26; ++x) {
        int32_t acc = L.bconv[oc];
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            acc = mac(L.image[(y + ky) * 28 + (x + kx)], -12, 32,
                      L.wconv[oc * 9 + ky * 3 + kx], -12, 32, acc, -12, st);
          }
        }
        if (acc < 0) {
          acc = 0;
          st.relu_zeros++;
        }
        L.conv[(y * 26 + x) * 4 + oc] = acc;
      }
    }
  }
  for (int y = 0; y < 13; ++y) {
    for (int x = 0; x < 13; ++x) {
      for (int oc = 0; oc < 4; ++oc) {
        int32_t best = L.conv[((y * 2) * 26 + (x * 2)) * 4 + oc];
        best = std::max(best, L.conv[((y * 2) * 26 + (x * 2 + 1)) * 4 + oc]);
        best = std::max(best, L.conv[((y * 2 + 1) * 26 + (x * 2)) * 4 + oc]);
        best = std::max(best, L.conv[((y * 2 + 1) * 26 + (x * 2 + 1)) * 4 + oc]);
        L.pool[(y * 13 + x) * 4 + oc] = best;
      }
    }
  }
  for (int j = 0; j < 10; ++j) {
    int32_t acc = L.bfc[j] = quantize(m.bfc[j], -12, 32, st);
    for (int i = 0; i < kFlat; ++i)
      acc = mac(L.pool[i], -12, 32, L.wfc[i * 10 + j], -12, 32, acc, -12, st);
    L.logits[j] = acc;
  }
}

int32_t mul_raw_m12(int32_t a, int32_t b) {
  spirv::_detail::fxp::FxpBinaryFuncOpTy lhs = {uint32_t(a)};
  spirv::_detail::fxp::FxpBinaryFuncOpTy rhs = {uint32_t(b)};
  spirv::_detail::fxp::FxpBinaryFuncOpTy prod = {0};
  int exp = 0;
  spirv::_detail::fxp::binary_mul(4, lhs, -12, 4, rhs, -12, 4, prod, exp);
  if (exp != -24) return 0x7fffffff;
  return int32_t(prod.Full);
}

void write_i32(FILE* f, const int32_t* p, int n) {
  if (std::fwrite(p, sizeof(int32_t), size_t(n), f) != size_t(n))
    std::exit(1);
}

int export_sim(const char* data, const char* weights, const char* case_path, int index) {
  Model model;
  if (!load_weights(weights, model)) return 1;
  std::vector<double> images;
  std::vector<int> labels;
  if (!load_images(std::string(data) + "/t10k-images-idx3-ubyte", images) ||
      !load_labels(std::string(data) + "/t10k-labels-idx1-ubyte", labels))
    return 1;
  if (index < 0 || index >= 10000) return 1;
  Layers32 L;
  Stats st;
  forward_fxp32_layers(model, &images[index * 784], L, st);
  double yf[10], y32[10];
  forward_float(model, &images[index * 784], yf);
  for (int j = 0; j < 10; ++j)
    y32[j] = double(L.logits[j]) * std::ldexp(1.0, -12);
  const int32_t probe_a[] = {1, 2, 3, 4};
  const int32_t probe_b[] = {-3, -3, -3, -3};
  const int32_t req_a[] = {
      -100, 100, -100, 2048, -2048, -2047, 2047, INT32_MIN, 3000, -3000,
      4095, -4095, 4096, -4096, 1, -1};
  const int32_t req_b[] = {
      50, -50, -50, 1, 1, 1, 1, 1, -3000, 3000,
      1, 1, 1, 1, -5000, 2048};
  int32_t probe_e[4], req_e[16];
  for (int i = 0; i < 4; ++i) probe_e[i] = mul_raw_m12(probe_a[i], probe_b[i]);
  for (int i = 0; i < 16; ++i) {
    const int32_t raw = mul_raw_m12(req_a[i], req_b[i]);
    req_e[i] = int32_t(fxp_requantize(raw, -24, -12, 32));
  }
  FILE* f = std::fopen(case_path, "wb");
  if (!f) return 1;
  const int32_t hdr[] = {
      int32_t(0x31435058), index, labels[index], argmax10(yf), argmax10(y32),
      4, 16, 26 * 26 * 4, kFlat, 10};
  write_i32(f, hdr, 10);
  write_i32(f, probe_a, 4);
  write_i32(f, probe_b, 4);
  write_i32(f, probe_e, 4);
  write_i32(f, req_a, 16);
  write_i32(f, req_b, 16);
  write_i32(f, req_e, 16);
  write_i32(f, L.image, 784);
  write_i32(f, L.wconv, 36);
  write_i32(f, L.bconv, 4);
  write_i32(f, L.conv, 26 * 26 * 4);
  write_i32(f, L.pool, kFlat);
  write_i32(f, L.wfc, kFlat * 10);
  write_i32(f, L.bfc, 10);
  write_i32(f, L.logits, 10);
  std::fclose(f);
  std::printf("EXPORT index %d label %d cpu_float_argmax %d cpu_fxp_argmax %d clips %llu overflow %llu conv0 %d logit0 %d\n",
              index, labels[index], argmax10(yf), argmax10(y32),
              st.convert_clips, st.product_overflows, L.conv[0], L.logits[0]);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 6 && std::strcmp(argv[1], "--export-sim") == 0)
    return export_sim(argv[2], argv[3], argv[4], std::atoi(argv[5]));
  if (argc != 4) {
    std::fprintf(stderr, "usage: eval_fxp_cnn DATA_DIR WEIGHTS OUT_JSON\n"
                         "       eval_fxp_cnn --export-sim DATA WEIGHTS CASE INDEX\n");
    return 2;
  }
  Model model;
  if (!load_weights(argv[2], model)) {
    std::fprintf(stderr, "bad weights\n");
    return 1;
  }
  std::vector<double> images;
  std::vector<int> labels;
  const std::string data = argv[1];
  if (!load_images(data + "/t10k-images-idx3-ubyte", images) ||
      !load_labels(data + "/t10k-labels-idx1-ubyte", labels)) {
    std::fprintf(stderr, "bad mnist\n");
    return 1;
  }
  int float_ok = 0, fxp32_ok = 0, fxp8_ok = 0;
  double mae32 = 0, mae8 = 0;
  Stats st32, st8;
  for (int n = 0; n < 10000; ++n) {
    double yf[10], y32[10], y8[10];
    forward_float(model, &images[n * 784], yf);
    forward_fxp(model, &images[n * 784], y32, -12, 32, -12, st32);
    forward_fxp(model, &images[n * 784], y8, -4, 8, -8, st8);
    if (argmax10(yf) == labels[n]) float_ok++;
    if (argmax10(y32) == labels[n]) fxp32_ok++;
    if (argmax10(y8) == labels[n]) fxp8_ok++;
    for (int j = 0; j < 10; ++j) {
      mae32 += std::fabs(y32[j] - yf[j]);
      mae8 += std::fabs(y8[j] - yf[j]);
    }
    if ((n + 1) % 2000 == 0)
      std::fprintf(stderr, "cnn images %d/10000\n", n + 1);
  }
  mae32 /= 100000.0;
  mae8 /= 100000.0;
  FILE* out = std::fopen(argv[3], "w");
  if (!out) return 1;
  std::fprintf(out,
    "{\n"
    "  \"network\": \"conv3x3x4-relu-pool2-fc10\",\n"
    "  \"paper_mlp\": false,\n"
    "  \"test_images\": 10000,\n"
    "  \"float_correct\": %d,\n"
    "  \"fxp32_correct\": %d,\n"
    "  \"fxp8_correct\": %d,\n"
    "  \"mae_fxp32_logits\": %.8g,\n"
    "  \"mae_fxp8_logits\": %.8g,\n"
    "  \"fxp32_product_overflows\": %llu,\n"
    "  \"fxp8_product_overflows\": %llu,\n"
    "  \"fxp32_convert_clips\": %llu,\n"
    "  \"fxp8_convert_clips\": %llu,\n"
    "  \"fxp32_relu_zeros\": %llu,\n"
    "  \"fxp8_relu_zeros\": %llu,\n"
    "  \"fxp32_macs\": %llu,\n"
    "  \"relu_and_pool\": \"integer compare and max on accumulator raw values, not parser opcodes\",\n"
    "  \"simulator\": false\n"
    "}\n",
    float_ok, fxp32_ok, fxp8_ok, mae32, mae8,
    st32.product_overflows, st8.product_overflows,
    st32.convert_clips, st8.convert_clips,
    st32.relu_zeros, st8.relu_zeros, st32.macs);
  std::fclose(out);
  std::printf("float %d fxp32 %d fxp8 %d / 10000\n", float_ok, fxp32_ok, fxp8_ok);
  return 0;
}
