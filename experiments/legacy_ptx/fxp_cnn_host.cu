/* Host for the bespoke fixed-CNN PTX. Compares raw tensors to the CPU case.
   Power and timing numbers from GPGPU-Sim are not read. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Dummy bodies. PTX_SIM_KERNELFILE replaces them with fxp_cnn_image.ptx. */
extern "C" __global__ void kprobe(const int *a, const int *b, int *out, int n) {
  if (n > 0) out[0] = a[0] + b[0];
}
extern "C" __global__ void kreq(const int *a, const int *b, int *out, int n) {
  if (n > 0) out[0] = a[0] + b[0];
}
extern "C" __global__ void ktile(const int *img, const int *w, const int *bias, int *out) {
  out[0] = img[0] + w[0] + bias[0];
}
extern "C" __global__ void kconv(const int *img, const int *w, const int *bias, int *out) {
  out[0] = img[0] + w[0] + bias[0];
}
extern "C" __global__ void kpool(const int *src, int *dst) {
  dst[0] = src[0];
}
extern "C" __global__ void kfc(const int *act, const int *w, const int *bias, int *out) {
  out[0] = act[0] + w[0] + bias[0];
}

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int fail(const char *msg) {
  printf("SIMCNN fail %s\n", msg);
  return 2;
}

static int *upload(const int *src, int n, const char *what) {
  int *d = NULL;
  cudaError_t e = cudaMalloc((void **)&d, sizeof(int) * (size_t)n);
  if (e != cudaSuccess) {
    printf("SIMCNN cudaMalloc %s %s\n", what, cudaGetErrorString(e));
    return NULL;
  }
  e = cudaMemcpy(d, src, sizeof(int) * (size_t)n, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    printf("SIMCNN cudaMemcpy %s %s\n", what, cudaGetErrorString(e));
    return NULL;
  }
  return d;
}

static int download(int *dst, int *src, int n, const char *what) {
  cudaError_t e = cudaMemcpy(dst, src, sizeof(int) * (size_t)n, cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    printf("SIMCNN cudaMemcpyD2H %s %s\n", what, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}

static int mismatches(const int *got, const int *exp, int n) {
  int bad = 0;
  int shown = 0;
  for (int i = 0; i < n; ++i) {
    if (got[i] != exp[i]) {
      if (shown < 4) {
        printf("SIMCNN mismatch_at %d got %d expect %d\n", i, got[i], exp[i]);
        shown++;
      }
      bad++;
    }
  }
  return bad;
}

static void layer_begin(const char *name) {
  printf("SIMCNN LAYER %s begin\n", name);
  fflush(stdout);
}

static void layer_end(const char *name, double sec) {
  printf("SIMCNN LAYER %s end seconds %.6f\n", name, sec);
  fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: fxp_cnn_host CASE\n");
    return 2;
  }
  FILE *f = fopen(argv[1], "rb");
  if (!f) return fail("open");
  int hdr[10];
  if (fread(hdr, 4, 10, f) != 10) return fail("header");
  if (hdr[0] != 0x31435058) return fail("magic");
  const int index = hdr[1];
  const int label = hdr[2];
  const int float_argmax = hdr[3];
  const int fxp_argmax = hdr[4];
  const int nprobe = hdr[5];
  const int nreq = hdr[6];
  const int nconv = hdr[7];
  const int npool = hdr[8];
  const int nlogit = hdr[9];
  if (nprobe != 4 || nreq != 16 || nconv != 26 * 26 * 4 || npool != 676 || nlogit != 10)
    return fail("shape");
  int *probe_a = (int *)malloc(sizeof(int) * 4);
  int *probe_b = (int *)malloc(sizeof(int) * 4);
  int *probe_e = (int *)malloc(sizeof(int) * 4);
  int *req_a = (int *)malloc(sizeof(int) * 16);
  int *req_b = (int *)malloc(sizeof(int) * 16);
  int *req_e = (int *)malloc(sizeof(int) * 16);
  int *image = (int *)malloc(sizeof(int) * 784);
  int *wconv = (int *)malloc(sizeof(int) * 36);
  int *bconv = (int *)malloc(sizeof(int) * 4);
  int *conv = (int *)malloc(sizeof(int) * (size_t)nconv);
  int *pool = (int *)malloc(sizeof(int) * (size_t)npool);
  int *wfc = (int *)malloc(sizeof(int) * 6760);
  int *bfc = (int *)malloc(sizeof(int) * 10);
  int *logits = (int *)malloc(sizeof(int) * 10);
  if (fread(probe_a, 4, 4, f) != 4 || fread(probe_b, 4, 4, f) != 4 ||
      fread(probe_e, 4, 4, f) != 4 || fread(req_a, 4, 16, f) != 16 ||
      fread(req_b, 4, 16, f) != 16 || fread(req_e, 4, 16, f) != 16 ||
      fread(image, 4, 784, f) != 784 || fread(wconv, 4, 36, f) != 36 ||
      fread(bconv, 4, 4, f) != 4 || fread(conv, 4, (size_t)nconv, f) != (size_t)nconv ||
      fread(pool, 4, (size_t)npool, f) != (size_t)npool ||
      fread(wfc, 4, 6760, f) != 6760 || fread(bfc, 4, 10, f) != 10 ||
      fread(logits, 4, 10, f) != 10) {
    return fail("payload");
  }
  fclose(f);
  printf("SIMCNN case index %d label %d cpu_float_argmax %d cpu_fxp_argmax %d\n",
         index, label, float_argmax, fxp_argmax);
  fflush(stdout);

  int *d_pa = upload(probe_a, 4, "probe_a");
  int *d_pb = upload(probe_b, 4, "probe_b");
  int *d_po = upload(probe_e, 4, "probe_out");
  int *d_ra = upload(req_a, 16, "req_a");
  int *d_rb = upload(req_b, 16, "req_b");
  int *d_ro = upload(req_e, 16, "req_out");
  int *d_img = upload(image, 784, "image");
  int *d_wc = upload(wconv, 36, "wconv");
  int *d_bc = upload(bconv, 4, "bconv");
  int *d_conv = upload(conv, nconv, "conv");
  int *d_pool = upload(pool, npool, "pool");
  int *d_wf = upload(wfc, 6760, "wfc");
  int *d_bf = upload(bfc, 10, "bfc");
  int *d_log = upload(logits, 10, "logits");
  int *d_tile = upload(conv, 1, "tile");
  if (!d_pa || !d_pb || !d_po || !d_ra || !d_rb || !d_ro || !d_img || !d_wc ||
      !d_bc || !d_conv || !d_pool || !d_wf || !d_bf || !d_log || !d_tile)
    return fail("upload");

  int got_probe[4], got_req[16], got_tile, *got_conv, *got_pool, got_log[10];
  got_conv = (int *)malloc(sizeof(int) * (size_t)nconv);
  got_pool = (int *)malloc(sizeof(int) * (size_t)npool);
  int bad = 0;
  double t0, t1;

  layer_begin("probe");
  t0 = now_sec();
  cudaMemset(d_po, 0x7f, sizeof(int) * 4);
  kprobe<<<1, 1>>>(d_pa, d_pb, d_po, nprobe);
  if (cudaDeviceSynchronize() != cudaSuccess) return fail("probe_sync");
  t1 = now_sec();
  layer_end("probe", t1 - t0);
  if (download(got_probe, d_po, 4, "probe")) return fail("probe_copy");
  int pb = mismatches(got_probe, probe_e, 4);
  printf("SIMCNN probe_mismatch %d\n", pb);
  bad += pb;
  if (pb) return fail("probe");

  layer_begin("requant");
  t0 = now_sec();
  cudaMemset(d_ro, 0x7f, sizeof(int) * 16);
  kreq<<<1, 1>>>(d_ra, d_rb, d_ro, nreq);
  if (cudaDeviceSynchronize() != cudaSuccess) return fail("req_sync");
  t1 = now_sec();
  layer_end("requant", t1 - t0);
  if (download(got_req, d_ro, 16, "req")) return fail("req_copy");
  int rb = mismatches(got_req, req_e, 16);
  printf("SIMCNN requant_mismatch %d\n", rb);
  bad += rb;
  if (rb) return fail("requant");

  layer_begin("tile");
  t0 = now_sec();
  cudaMemset(d_tile, 0x7f, sizeof(int));
  ktile<<<1, 1>>>(d_img, d_wc, d_bc, d_tile);
  if (cudaDeviceSynchronize() != cudaSuccess) return fail("tile_sync");
  t1 = now_sec();
  layer_end("tile", t1 - t0);
  if (download(&got_tile, d_tile, 1, "tile")) return fail("tile_copy");
  printf("SIMCNN tile got %d expect %d\n", got_tile, conv[0]);
  if (got_tile != conv[0]) return fail("tile");

  layer_begin("conv");
  t0 = now_sec();
  cudaMemset(d_conv, 0x7f, sizeof(int) * (size_t)nconv);
  kconv<<<1, 1>>>(d_img, d_wc, d_bc, d_conv);
  if (cudaDeviceSynchronize() != cudaSuccess) return fail("conv_sync");
  t1 = now_sec();
  layer_end("conv", t1 - t0);
  if (download(got_conv, d_conv, nconv, "conv")) return fail("conv_copy");
  int cb = mismatches(got_conv, conv, nconv);
  printf("SIMCNN conv_mismatch %d conv0 %d expect %d\n", cb, got_conv[0], conv[0]);
  bad += cb;
  if (cb) return fail("conv");

  layer_begin("pool");
  t0 = now_sec();
  cudaMemset(d_pool, 0x7f, sizeof(int) * (size_t)npool);
  kpool<<<1, 1>>>(d_conv, d_pool);
  if (cudaDeviceSynchronize() != cudaSuccess) return fail("pool_sync");
  t1 = now_sec();
  layer_end("pool", t1 - t0);
  if (download(got_pool, d_pool, npool, "pool")) return fail("pool_copy");
  int ob = mismatches(got_pool, pool, npool);
  printf("SIMCNN pool_mismatch %d pool0 %d expect %d\n", ob, got_pool[0], pool[0]);
  bad += ob;
  if (ob) return fail("pool");

  layer_begin("fc");
  t0 = now_sec();
  cudaMemset(d_log, 0x7f, sizeof(int) * 10);
  kfc<<<1, 1>>>(d_pool, d_wf, d_bf, d_log);
  if (cudaDeviceSynchronize() != cudaSuccess) return fail("fc_sync");
  t1 = now_sec();
  layer_end("fc", t1 - t0);
  if (download(got_log, d_log, 10, "logits")) return fail("fc_copy");
  int lb = mismatches(got_log, logits, 10);
  printf("SIMCNN logit_mismatch %d\n", lb);
  printf("SIMCNN sim_logits %d %d %d %d %d %d %d %d %d %d\n",
         got_log[0], got_log[1], got_log[2], got_log[3], got_log[4],
         got_log[5], got_log[6], got_log[7], got_log[8], got_log[9]);
  printf("SIMCNN cpu_fxp_logits %d %d %d %d %d %d %d %d %d %d\n",
         logits[0], logits[1], logits[2], logits[3], logits[4],
         logits[5], logits[6], logits[7], logits[8], logits[9]);
  fflush(stdout);
  int sim_arg = 0;
  for (int i = 1; i < 10; ++i)
    if (got_log[i] > got_log[sim_arg]) sim_arg = i;
  printf("SIMCNN sim_argmax %d cpu_fxp_argmax %d cpu_float_argmax %d label %d\n",
         sim_arg, fxp_argmax, float_argmax, label);
  bad += lb;
  printf("SIMCNN status %d\n", bad == 0 && sim_arg == fxp_argmax ? 0 : 1);
  fflush(stdout);
  return bad == 0 && sim_arg == fxp_argmax ? 0 : 2;
}
