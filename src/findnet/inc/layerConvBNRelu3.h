// flag_DW                        0
// out_mult                       22
// out_shift                      22
// FLAG_BATCHNORM                 1
// FLAG_RELU                      1
// weight_T                       0
// to_compute_acc_in              0
// to_compute_acc_out             0
// test_location                  L3
// tile_dim_h                     1
// tile_dim_w                     1
// optional                       conv
// conv_order                     PULP-NN
// type                           char
// func_name                      layerConvBNRelu3
// l1_x_offset                    0
// l1_y_offset                    14340
// l1_W_offset                    28680
// l1_k_offset                    37908
// l1_lambda_offset               37976
// k_size_byte                    64
// lambda_size_byte               128
// k_tile_size_byte               64
// lambda_tile_size_byte          128
// tile_dim_nof                   1
// tile_dim_nif                   1
// border                         1
// nof                            32
// nif                            32
// h                              16
// w                              28
// fs1                            3
// fs2                            3
// conv_overlap1                  2
// conv_overlap2                  2
// has_bias                       0
// padding                        1
// stride                         1
// x_h                            16
// x_w                            28
// x_data_size_byte               1
// x_tile_size_nif                32
// x_tile_size_h                  16
// x_tile_size_w                  28
// x_tile_size_byte               14336
// x_stride_w_byte                896
// x_stride_c_byte                32
// x_length_nif_px                32
// x_length_nif_byte              32
// x_length_h_px                  16
// x_length_w_byte                28
// x_tile_size_nif_last           32
// x_tile_size_h_last             16
// x_tile_size_w_last             28
// x_length_nif_px_last           32
// x_length_nif_byte_last         32
// x_length_h_px_last             16
// x_length_w_byte_last           28
// x_tile_size_byte_first         14336
// x_length_nif_px_first          32
// x_length_nif_byte_first        32
// x_length_h_px_first            16
// x_length_w_byte_first          28
// W_nof                          32
// b_tile_size_byte               32
// W_nif                          32
// W_data_size_byte               1
// W_tile_size_nof                32
// W_tile_size_nif                32
// W_tile_size_byte               9216
// W_stride_nof_byte              288
// W_stride_hw_byte               32
// W_length_nif_byte              32
// W_tile_size_nof_last           32
// W_tile_size_nif_last           32
// W_length_nif_byte_last         32
// W_tile_size_byte_first         9216
// W_length_nif_byte_first        32
// b_size_byte                    32
// l2_off_k                       9216
// l2_off_lambda                  9280
// y_h                            16
// y_w                            28
// y_data_size_byte               1
// y_tile_size_nof                32
// y_tile_size_h                  16
// y_tile_size_w                  28
// y_tile_size_byte               14336
// y_stride_w_byte                896
// y_stride_c_byte                32
// y_length_nof_px                32
// y_length_nof_byte              32
// y_length_h_px                  16
// y_length_w_byte                28
// y_tile_size_nof_last           32
// y_tile_size_h_last             16
// y_tile_size_w_last             28
// y_length_nof_px_last           32
// y_length_w_byte_last           28
// y_length_nof_byte_last         32


#include "pulp.h"
#include "dory.h"
#include "stats.h"

void  layerConvBNRelu3(
  unsigned int l2_x,
  unsigned int l2_y,
  unsigned int l2_W,
  unsigned int l1_buffer
);