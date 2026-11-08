#ifndef FXP_POWER_POLICY_H
#define FXP_POWER_POLICY_H

// Fixed-point activity has no measured McPAT ALU and no bit-width scale.
// If the XML does not name FXP_INT, one FXP instruction uses the same
// activity coefficient as one integer instruction (the stock FP_INT value).
inline double fxp_activity_coefficient(int xml_specified, double xml_value,
                                       double integer_coefficient) {
  if (xml_specified)
    return xml_value;
  return integer_coefficient;
}

#endif
