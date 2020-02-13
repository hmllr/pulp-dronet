// flag_DW                        0
// out_mult                       504
// out_shift                      9
// FLAG_BATCHNORM                 0
// FLAG_RELU                      1
// weight_T                       0
// to_compute_acc_in              0
// to_compute_acc_out             0
// test_location                  L3
// tile_dim_h                     1
// tile_dim_w                     1
// optional                       AveragePoolRelu
// conv_order                     PULP-NN-MAX
// type                           char
// func_name                      layerAveragePoolRelu8
// l1_x_offset                    0
// l1_y_offset                    3588
// tile_dim_nof                   1
// tile_dim_nif                   1
// border                         0
// nof                            128
// nif                            128
// h                              4
// w                              7
// fs1                            4
// fs2                            7
// conv_overlap1                  3
// conv_overlap2                  6
// has_bias                       1
// padding                        0
// stride                         1
// x_h                            4
// x_w                            7
// x_data_size_byte               1
// x_tile_size_nif                128
// x_tile_size_h                  4
// x_tile_size_w                  7
// x_tile_size_byte               3584
// x_stride_w_byte                896
// x_stride_c_byte                128
// x_length_nif_px                128
// x_length_nif_byte              128
// x_length_h_px                  4
// x_length_w_byte                7
// x_tile_size_nif_last           128
// x_tile_size_h_last             4
// x_tile_size_w_last             7
// x_length_nif_px_last           128
// x_length_nif_byte_last         128
// x_length_h_px_last             4
// x_length_w_byte_last           7
// x_tile_size_byte_first         3584
// x_length_nif_px_first          128
// x_length_nif_byte_first        128
// x_length_h_px_first            4
// x_length_w_byte_first          7
// y_h                            1
// y_w                            1
// y_data_size_byte               1
// y_tile_size_nof                128
// y_tile_size_h                  1
// y_tile_size_w                  1
// y_tile_size_byte               128
// y_stride_w_byte                128
// y_stride_c_byte                128
// y_length_nof_px                128
// y_length_nof_byte              128
// y_length_h_px                  1
// y_length_w_byte                1
// y_tile_size_nof_last           128
// y_tile_size_h_last             1
// y_tile_size_w_last             1
// y_length_nof_px_last           128
// y_length_w_byte_last           1
// y_length_nof_byte_last         128


#include "pulp.h"
#include "dory.h"
#include "stats.h"

void  layerAveragePoolRelu8(
  unsigned int l2_x,
  unsigned int l2_y,
  unsigned int l1_buffer
);