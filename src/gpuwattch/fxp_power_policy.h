#ifndef FXP_POWER_POLICY_H
#define FXP_POWER_POLICY_H

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

#endif
