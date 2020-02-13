// flag_DW                        0
// out_mult                       17
// out_shift                      23
// FLAG_BATCHNORM                 1
// FLAG_RELU                      1
// weight_T                       0
// to_compute_acc_in              0
// to_compute_acc_out             0
// test_location                  L3
// tile_dim_h                     4
// tile_dim_w                     2
// optional                       conv
// conv_order                     PULP-NN
// type                           char
// func_name                      layerConvBNRelu6
// l1_x_offset                    0
// l1_y_offset                    3460
// l1_W_offset                    3656
// l1_k_offset                    31316
// l1_lambda_offset               31576
// k_size_byte                    256
// lambda_size_byte               512
// k_tile_size_byte               48
// lambda_tile_size_byte          96
// tile_dim_nof                   6
// tile_dim_nif                   1
// border                         1
// nof                            128
// nif                            64
// h                              8
// w                              14
// fs1                            3
// fs2                            3
// conv_overlap1                  1
// conv_overlap2                  1
// has_bias                       0
// padding                        1
// stride                         2
// x_h                            8
// x_w                            14
// x_data_size_byte               1
// x_tile_size_nif                64
// x_tile_size_h                  3
// x_tile_size_w                  9
// x_tile_size_byte               1728
// x_stride_w_byte                896
// x_stride_c_byte                64
// x_length_nif_px                64
// x_length_nif_byte              64
// x_length_h_px                  3
// x_length_w_byte                9
// x_tile_size_nif_last           64
// x_tile_size_h_last             3
// x_tile_size_w_last             7
// x_length_nif_px_last           64
// x_length_nif_byte_last         64
// x_length_h_px_last             3
// x_length_w_byte_last           7
// x_tile_size_byte_first         1728
// x_length_nif_px_first          64
// x_length_nif_byte_first        64
// x_length_h_px_first            3
// x_length_w_byte_first          9
// W_nof                          128
// b_tile_size_byte               24
// W_nif                          64
// W_data_size_byte               1
// W_tile_size_nof                24
// W_tile_size_nif                64
// W_tile_size_byte               13824
// W_stride_nof_byte              576
// W_stride_hw_byte               64
// W_length_nif_byte              64
// W_tile_size_nof_last           8
// W_tile_size_nif_last           64
// W_length_nif_byte_last         64
// W_tile_size_byte_first         13824
// W_length_nif_byte_first        64
// b_size_byte                    128
// l2_off_k                       73728
// l2_off_lambda                  73984
// y_h                            4
// y_w                            7
// y_data_size_byte               1
// y_tile_size_nof                24
// y_tile_size_h                  1
// y_tile_size_w                  4
// y_tile_size_byte               96
// y_stride_w_byte                896
// y_stride_c_byte                128
// y_length_nof_px                24
// y_length_nof_byte              24
// y_length_h_px                  1
// y_length_w_byte                4
// y_tile_size_nof_last           8
// y_tile_size_h_last             1
// y_tile_size_w_last             3
// y_length_nof_px_last           8
// y_length_w_byte_last           3
// y_length_nof_byte_last         8


#include "pulp.h"
#include "dory.h"
#include "stats.h"

void  layerConvBNRelu6(
  unsigned int l2_x,
  unsigned int l2_y,
  unsigned int l2_W,
  unsigned int l1_buffer
);