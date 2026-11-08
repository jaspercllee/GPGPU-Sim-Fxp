#ifndef FXP_POWER_POLICY_H
#define FXP_POWER_POLICY_H

#include <string.h>

// Uncalibrated compute proxy. Not a silicon measurement and not a bit-width scale.
// add/sub/convert use the integer ALU weight already applied to INT__OP (25).
// mul uses the 32-bit integer multiply weight (4). div uses the integer
// divide weight (40). McPAT has one MUL functional unit; GPGPU-Sim feeds both
// imul32 and idiv into that unit through the SFU access bundle. Generic
// INT_MUL is historically summed into the FPU counter and is not used here.
// The explicit scale multiplies those stock weights. Zero adds no accesses.
enum {
  FXP_PROXY_NONE = 0,
  FXP_PROXY_ADD = 1,
  FXP_PROXY_SUB = 2,
  FXP_PROXY_MUL = 3,
  FXP_PROXY_DIV = 4,
  FXP_PROXY_CVT = 5
};

enum {
  FXP_PROXY_UNIT_NONE = 0,
  FXP_PROXY_UNIT_ALU = 1,
  FXP_PROXY_UNIT_MUL = 2
};

inline int fxp_proxy_class_from_name(const char *fname) {
  if (!fname)
    return FXP_PROXY_NONE;
  if (strstr(fname, "fxp_add"))
    return FXP_PROXY_ADD;
  if (strstr(fname, "fxp_sub"))
    return FXP_PROXY_SUB;
  if (strstr(fname, "fxp_mul"))
    return FXP_PROXY_MUL;
  if (strstr(fname, "fxp_div"))
    return FXP_PROXY_DIV;
  if (strstr(fname, "fxp") && strstr(fname, "convert"))
    return FXP_PROXY_CVT;
  return FXP_PROXY_NONE;
}

inline int fxp_proxy_unit(int cls) {
  if (cls == FXP_PROXY_ADD || cls == FXP_PROXY_SUB || cls == FXP_PROXY_CVT)
    return FXP_PROXY_UNIT_ALU;
  if (cls == FXP_PROXY_MUL || cls == FXP_PROXY_DIV)
    return FXP_PROXY_UNIT_MUL;
  return FXP_PROXY_UNIT_NONE;
}

// Stock activity weights from incexecstat. Not introduced for fixed point.
inline double fxp_proxy_stock_weight(int cls) {
  if (cls == FXP_PROXY_ADD || cls == FXP_PROXY_SUB || cls == FXP_PROXY_CVT)
    return 25.0;
  if (cls == FXP_PROXY_MUL)
    return 4.0;
  if (cls == FXP_PROXY_DIV)
    return 40.0;
  return 0.0;
}

inline double fxp_proxy_latency(int cls, double scale) {
  if (scale == 0.0)
    return 0.0;
  return fxp_proxy_stock_weight(cls) * scale;
}

// The stock FP_INT coefficient scales instruction-window activity.
// It is not a measured fixed-point ALU energy and it is not a bit-width scale.
// Copy it only when the XML does not name FXP_INT.
inline double fxp_activity_coefficient(int xml_specified, double xml_value,
                                       double integer_coefficient) {
  if (xml_specified)
    return xml_value;
  return integer_coefficient;
}

// print_power_kernel_stats writes GPUenergy*1e12. GPUenergy only increases.
// reset_counters() does not clear it, so each kernel print is a running total.
// The run total is the last print. Adding the prints counts earlier samples again.
inline double last_cumulative_joules(const double *cumulative_pj, int n) {
  if (n <= 0)
    return 0.0;
  return cumulative_pj[n - 1] * 1e-12;
}

inline double summed_cumulative_joules(const double *cumulative_pj, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += cumulative_pj[i] * 1e-12;
  return sum;
}

// Default 1: one fixed-point op gets the stock integer weight of its class.
// 0 adds no execution-unit accesses. Not a calibrated uncertainty.
extern double g_fxp_compute_proxy_scale;

#endif
