/*
 * pulp_nn_quant_u8.c
 * Nazareno Bruschi <nazareno.bruschi@studio.unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 * Francesco Conti <f.conti@unibo.it>
 * 
 * Copyright (C) 2019 ETH Zurich, University of Bologna.
 * All rights reserved.
 */

#include "rt/rt_api.h"
#include "utils.h"
#include "kernels.h"

#define clip8(x) __builtin_pulp_clipu_r(x, 255)

uint8_t __attribute__((always_inline)) pulp_nn_quant_u8(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  /* Quantization */
  int16_t x = (m * phi) >> d;
  uint8_t res = clip8(x);
  return res;
}
