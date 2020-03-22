// flag_DW                        0
// out_mult                       24
// out_mult2                      0
// out_shift                      24
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


#include "layerConvBNRelu6.h"
//DMA events
extern   unsigned int dma_read_evt_W;
extern   unsigned int dma_read_evt_x;
extern   unsigned int dma_write_evt_y;
extern   unsigned int dma_read_evt_lambda;
extern   unsigned int dma_read_evt_k;
extern   int p_r, p_l, p_t, p_b;
extern   int last_nof_exec;
extern   int last_nif_exec;
extern   int last_h_exec;
extern   int last_w_exec;
extern   unsigned short x_tile_size_nif;
extern   unsigned short  x_tile_size_h;
extern   unsigned short  x_tile_size_w;
extern   unsigned short  x_tile_size_byte;
extern   unsigned short  x_length_h_px;
extern   unsigned short  x_length_nif_byte;
extern   int pad_offset_h, pad_offset_w;
extern   unsigned short  W_tile_size_nof;
extern   unsigned short  W_tile_size_nif;
extern   unsigned short  W_tile_size_byte;
extern   unsigned short W_length_nif_byte;
extern   char *x;
extern   char *W;
extern   char *y;
extern   char *b;
extern int16_t *k;
extern int32_t *lambda;
extern   int x_tile_size_nif_exec;
extern   int x_tile_size_h_exec;
extern   int x_tile_size_w_exec;
extern   int y_tile_size_nof;
extern   int y_tile_size_h;
extern   int y_tile_size_w;
extern   int y_tile_size_byte;
extern   int y_length_h_px;
extern   int y_length_nof_byte;
// compute double buffering offsets and update db state
extern   int db_x;
extern   int db_W;
extern   int db_y;
extern   int exec_db_x;
extern   int exec_db_W;

extern char *im2col;
  
/*
l2_x --> activations input + activations accumulated
l2_y --> activations output + activations output accumulated
l2_W --> weights + k + lambda
*/
void layerConvBNRelu6(
  unsigned int l2_x,
  unsigned int l2_y,
  unsigned int l2_W,
  unsigned int l1_buffer,
  unsigned int out_mult_in,
  unsigned int out_shift_in
) {

  if(rt_core_id()==0){
    im2col = l1_buffer + 32144;
    // copy first tiles
    //l2_x has now activations, input activations, accumulated activations over channels
    dory_dma_memcpy_3d_custom(
    l2_x, // ext
    (l1_buffer + 0) + 0, // loc
    1728, // size: dimension of the buffer
    896, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
    64, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
    3,// length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
    64, // length_0: legnth of the 1_d copy, the length of tile in w direction
    1, // dir
    &dma_read_evt_x // copy
    );
    dory_dma_memcpy_3d_custom(
    l2_W, // ext
    (l1_buffer + 3656) + 0, // loc offset caused by size of tile_x*2 (double_buffer) and tile_y*2 (double buffer)
    13824, // size: dimension of matrix of weight * bytes_per_weight
    576, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
    64, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
    24, // length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
    64, // length_0: legnth of the 1_d copy, the length of tile in w direction
    1, // dir
    &dma_read_evt_W // copy
    );
    rt_dma_memcpy(
    l2_W+73728, // ext
    l1_buffer + 31316, // loc
    256, // size
    RT_DMA_DIR_EXT2LOC, // dir
    0, // merge
    &dma_read_evt_k // copy
    );
    rt_dma_memcpy(
    l2_W+73984, // ext
    l1_buffer + 31576, // loc
    512, // size
    RT_DMA_DIR_EXT2LOC, // dir
    0, // merge
    &dma_read_evt_lambda // copy
    );
  // bias is not double buffered
    rt_dma_wait(&dma_read_evt_k);
    rt_dma_wait(&dma_read_evt_lambda);

    // wait for x,W read
    mchan_barrier(dma_read_evt_x);
    mchan_free(dma_read_evt_x);
    mchan_barrier(dma_read_evt_W);
    mchan_free(dma_read_evt_W);
  }
  // tile loop indeces
    int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
    int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;
    int has_bias = 0;

    uint16_t out_mult = out_mult_in;
    uint16_t out_shift = out_shift_in;

  // double buffering state
    int db_state_x=0;
    int db_state_acc_in=0;
    int db_state_W=0;
    int db_state_y=1;
    int db_state_acc_out=1;

    int flag_first_ch_in;
    int flag_first_ch_out;
    int flag_last_ch_in;

  // last-tile flags
    int last_nof_load = (6 == 1) ? 1 : 0;
    int last_nif_load = (1 == 1) ? 1 : 0;
    int last_h_load = (4 == 1) ? 1 : 0;
    int last_w_load = (2 == 1) ? 1 : 0;

    int iter;
  // tile loop nest
  for(iter=0; iter<6*1*4*2; iter++) {
  if(rt_core_id()==0){
    // loop nest is nof,h,w,(nif=0)
    _i_w_load += 1;
    if(_i_w_load==2) {
      _i_w_load = 0;
      _i_h_load += 1;
      if(_i_h_load==4) {
        _i_h_load = 0;
        _i_nof_load += 1;
      }
    }

    if (_i_nif_exec==0)
      flag_first_ch_in = 1;
    else
      flag_first_ch_in = 0;
    if (_i_nof_exec==0)
      flag_first_ch_out = 1;
    else
      flag_first_ch_out = 0;

    flag_last_ch_in = 1;


    // wait for x,W read
    mchan_barrier(dma_read_evt_x);
    mchan_free(dma_read_evt_x);
    mchan_barrier(dma_read_evt_W);
    mchan_free(dma_read_evt_W);
    // check if last in any dimension
    last_nof_exec = last_nof_load;
    last_nif_exec = last_nif_load;
    last_h_exec = last_h_load;
    last_w_exec = last_w_load;
    last_nof_load = (_i_nof_load+1 == 6) ? 1 : 0;
    last_nif_load = (_i_nif_load+1 == 1) ? 1 : 0;
    last_h_load = (_i_h_load+1 == 4) ? 1 : 0;
    last_w_load = (_i_w_load+1 == 2) ? 1 : 0;

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? 1728 : 0;
    db_W = !db_state_W ? 13824 : 0;
    db_y = !db_state_y ? 96 : 0;
    exec_db_x = db_state_x ? 1728 : 0;
    db_state_x = ! db_state_x;
    exec_db_W = db_state_W ? 13824 : 0;
    if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      db_state_W = ! db_state_W;
    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single fil
    if(last_nif_exec) {
      db_state_y = ! db_state_y;
    }
    // double buffered reads
    if(iter<6*1*4*2-1) {
      x_tile_size_nif = (last_nif_load) ? 64 : 64;
      x_tile_size_h   = (last_h_load)   ? 3 : 3;
      x_tile_size_w   = (last_w_load)   ? 7 : 9;
      x_tile_size_byte = x_tile_size_nif*x_tile_size_h*x_tile_size_w*1;
      x_length_h_px = (last_h_load) ? 3 : 3;
      x_length_nif_byte = (last_nif_load)   ? 64 : 64;
      // additionally overlap by padding for the first tile after a border one
      //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      pad_offset_h=0, pad_offset_w=0;
      if(_i_h_load > 0)
        pad_offset_h = 1;
      if(_i_w_load > 0)
        pad_offset_w = 1;

      dory_dma_memcpy_3d_custom(
        dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, 3, 9, 64, 14, 64,  1, 1,0, pad_offset_h, pad_offset_w, 0, 1), // extern
        (l1_buffer + 0) + db_x, // loc
        x_tile_size_byte, // size: dimension of the buffer
        896, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
        64, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
        x_length_h_px,// length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
        x_length_nif_byte, // length_0: legnth of the 1_d copy, the length of tile in w direction
        1, // dir
        &dma_read_evt_x // copy
        );
      y_tile_size_h   = (last_h_load)   ? 1 : 1;
      y_tile_size_w   = (last_w_load)   ? 3 : 4;
      W_tile_size_nof = (last_nof_load) ? 8 : 24;
      W_tile_size_nif = (last_nif_load) ? 64 : 64;
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*1*3*3;
      W_length_nif_byte = (last_nif_load) ? 64 : 64;

      if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
        dory_dma_memcpy_3d_custom(
          dory_get_tile_3d(l2_W, _i_nof_load, 0, _i_nif_load, 24, 3*3, 64, 3*3, 64, 0,0,0,0,0,0, 1), // ext
          (l1_buffer + 3656) + db_W, // loc
          W_tile_size_byte, // size: dimension of matrix of weight * bytes_per_weight
          576, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
          64, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
          W_tile_size_nof, // length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
          W_length_nif_byte, // length_0: legnth of the 1_d copy, the length of tile in w direction
          1, // dir
          &dma_read_evt_W // copy
          );
    }
    x = (char *) (l1_buffer + 0 + exec_db_x);
    k = (int16_t *) (l1_buffer + 31316 + _i_nof_exec*48);
    lambda = (int32_t *) (l1_buffer + 31576 + _i_nof_exec*96);
    W = (char *) (l1_buffer + 3656 + exec_db_W);
    y = (char *) (l1_buffer + 3460 + db_y);
    x_tile_size_nif_exec = (last_nif_exec) ? 64 : 64;
    x_tile_size_h_exec   = (last_h_exec)   ? 3 : 3;
    x_tile_size_w_exec   = (last_w_exec)   ? 7 : 9;

    y_tile_size_nof = (last_nof_exec) ? 8 : 24;
    y_tile_size_h   = (last_h_exec)   ? 1 : 1;
    y_tile_size_w   = (last_w_exec)   ? 3 : 4;
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*1;
    y_length_h_px = (last_h_exec) ? 1 : 1;
    y_length_nof_byte = (last_nof_exec)   ? 8 : 24;
    p_r = 0;
    p_l = 0;
    p_t = 0;
    p_b = 0;
    if (_i_h_exec == 0)
      p_t = 1;
    if (_i_w_exec == 0)
      p_l = 1;
    if (_i_h_exec == 4-1)
      p_b = 1;
    if (_i_w_exec == 2-1)
      p_r = 1;
  }
  rt_team_barrier();
  pulp_nn_conv_i8_u8(
    x,
    0,
    x_tile_size_w_exec,
    x_tile_size_h_exec,
    x_tile_size_nif_exec,
    W,
    0,
    y_tile_size_nof,
    3,
    3,
    p_t,
    p_b,
    p_l,
    p_r,
    2,
    2,
    NULL,
    0,
    out_shift,
    out_mult,
    y,
    0,
    y_tile_size_w,
    y_tile_size_h,
    k,
    lambda,
    im2col,
    NULL,
    1,
    1,
    0,
    0,
    flag_first_ch_in,
    flag_first_ch_out
    );


if(rt_core_id()==0){
    // wait for DMA write
      if(iter) {
        mchan_barrier(dma_write_evt_y);
        mchan_free(dma_write_evt_y);
      }
      dory_dma_memcpy_3d_custom(
        dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, 1, 4, 24, 7, 128, 0, 0, 0, 0, 0, 0, 1), // ext
        (l1_buffer + 3460) + db_y, // loc
        y_tile_size_byte, // size
        896, // stride_1
        128, // stride_0
        y_length_h_px, // length_2
        y_length_nof_byte, // length_0
        0, // dir
        &dma_write_evt_y // copy
      );
    // update prev iterators
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
  }
  }
    //STOP_PROFILING();
  // wait for final write
  if(rt_core_id()==0){
    mchan_barrier(dma_write_evt_y);
    mchan_free(dma_write_evt_y);
  }


}
