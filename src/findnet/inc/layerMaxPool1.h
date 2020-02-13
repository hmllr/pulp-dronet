// flag_DW                        0
// out_mult                       0
// out_shift                      0
// FLAG_BATCHNORM                 0
// FLAG_RELU                      0
// weight_T                       0
// to_compute_acc_in              0
// to_compute_acc_out             0
// test_location                  L3
// tile_dim_h                     2
// tile_dim_w                     2
// optional                       MaxPool
// conv_order                     PULP-NN-MAX
// type                           char
// func_name                      layerMaxPool1
// l1_x_offset                    0
// l1_y_offset                    34820
// tile_dim_nof                   1
// tile_dim_nif                   1
// border                         1
// nof                            32
// nif                            32
// h                              30
// w                              54
// fs1                            2
// fs2                            2
// conv_overlap1                  0
// conv_overlap2                  0
// has_bias                       1
// padding                        1
// stride                         2
// x_h                            30
// x_w                            54
// x_data_size_byte               1
// x_tile_size_nif                32
// x_tile_size_h                  16
// x_tile_size_w                  34
// x_tile_size_byte               17408
// x_stride_w_byte                1728
// x_stride_c_byte                32
// x_length_nif_px                32
// x_length_nif_byte              32
// x_length_h_px                  16
// x_length_w_byte                34
// x_tile_size_nif_last           32
// x_tile_size_h_last             15
// x_tile_size_w_last             21
// x_length_nif_px_last           32
// x_length_nif_byte_last         32
// x_length_h_px_last             15
// x_length_w_byte_last           21
// x_tile_size_byte_first         17408
// x_length_nif_px_first          32
// x_length_nif_byte_first        32
// x_length_h_px_first            16
// x_length_w_byte_first          34
// y_h                            16
// y_w                            28
// y_data_size_byte               1
// y_tile_size_nof                32
// y_tile_size_h                  8
// y_tile_size_w                  17
// y_tile_size_byte               4352
// y_stride_w_byte                896
// y_stride_c_byte                32
// y_length_nof_px                32
// y_length_nof_byte              32
// y_length_h_px                  8
// y_length_w_byte                17
// y_tile_size_nof_last           32
// y_tile_size_h_last             8
// y_tile_size_w_last             11
// y_length_nof_px_last           32
// y_length_w_byte_last           11
// y_length_nof_byte_last         32


#include "pulp.h"
#include "dory.h"
#include "stats.h"

void  layerMaxPool1(
  unsigned int l2_x,
  unsigned int l2_y,
  unsigned int l1_buffer
);