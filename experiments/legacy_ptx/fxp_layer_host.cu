#include <stdio.h>
__global__ void ksub(int *o) { o[0] = 0; }
__global__ void kdiv(int *o) { o[0] = 0; }
__global__ void kdot(int *o) { o[0] = 0; }

static int launch(void (*fn)(int*), const char* name, int expect) {
  int *d;
  int h = -1;
  cudaError_t e = cudaMalloc((void**)&d, sizeof(int));
  printf("%s malloc %s\n", name, cudaGetErrorString(e));
  fn<<<1,1>>>(d);
  e = cudaDeviceSynchronize();
  printf("%s sync %s\n", name, cudaGetErrorString(e));
  e = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%s copy %s value %d expect %d\n", name, cudaGetErrorString(e), h, expect);
  cudaFree(d);
  return h == expect ? 0 : 1;
}

int main() {
  int bad = 0;
  bad += launch(ksub, "sub", 5);
  bad += launch(kdiv, "div", 3);
  bad += launch(kdot, "dot", 20);
  printf("fxp_layer_status %d\n", bad);
  return bad == 0 ? 0 : 2;
}
