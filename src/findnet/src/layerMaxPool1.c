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


#include "layerMaxPool1.h"
//DMA events
extern  unsigned int dma_read_evt_x;
extern  unsigned int dma_write_evt_y;

extern  int p_r, p_l, p_t, p_b;
extern  int last_nof_exec;
extern  int last_nif_exec;
extern  int last_h_exec;
extern  int last_w_exec;

extern  unsigned short x_tile_size_nif;
extern  unsigned short  x_tile_size_h;
extern  unsigned short  x_tile_size_w;
extern  unsigned short  x_tile_size_byte;
extern  unsigned short  x_length_h_px;
extern  unsigned short  x_length_nif_byte;
extern  int pad_offset_h, pad_offset_w;

extern char *x;
extern char *y;

extern  int x_tile_size_nif_exec;
extern  int x_tile_size_h_exec;
extern  int x_tile_size_w_exec;
extern  int y_tile_size_nof;
extern  int y_tile_size_h;
extern  int y_tile_size_w;
extern  int y_tile_size_byte;


extern  int y_length_h_px;
extern  int y_length_nof_byte;


// compute double buffering offsets and update db state
extern  int db_x;
extern  int db_y;

extern  int exec_db_x;
extern  int exec_db_W;


extern char *im2col;
  
/*
l2_x --> activations input + activations accumulated
l2_y --> activations output + activations output accumulated
l2_W --> weights + k + lambda
*/
void layerMaxPool1(
  unsigned int l2_x,
  unsigned int l2_y,
  unsigned int l1_buffer
) {
  if(rt_core_id()==0){
    im2col = l1_buffer + 34896;
    // copy first tiles
    //l2_x has now activations, input activations, accumulated activations over channels
    dory_dma_memcpy_3d_custom(
    l2_x, // ext
    (l1_buffer + 0) + 0, // loc
    17408, // size: dimension of the buffer
    1728, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
    32, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
    16,// length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
    32, // length_0: legnth of the 1_d copy, the length of tile in w direction
    1, // dir
    &dma_read_evt_x // copy
    );
    // wait for x,W read
    mchan_barrier(dma_read_evt_x);
    mchan_free(dma_read_evt_x);
  }
  // tile loop indeces
   int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
   int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;

  // double buffering state
   int db_state_x=0;
   int db_state_y=1;
   int db_state_acc_out=1;
   int flag_first_ch_out;

  // last-tile flags
   int last_nof_load = (1 == 1) ? 1 : 0;
   int last_nif_load = (1 == 1) ? 1 : 0;
   int last_h_load = (2 == 1) ? 1 : 0;
   int last_w_load = (2 == 1) ? 1 : 0;

   int iter;
  // tile loop nest
  for(iter=0; iter<1*2*2; iter++) {
  if(rt_core_id()==0){
    // loop nest is nof,h,w,(nif=0)
    _i_w_load += 1;
    if(_i_w_load==2) {
      _i_w_load = 0;
      _i_h_load += 1;
      if(_i_h_load==2) {
        _i_h_load = 0;
        _i_nif_load += 1;
        _i_nof_load += 1;
      }
    }

    if (_i_nof_exec==0)
      flag_first_ch_out = 1;
    else
      flag_first_ch_out = 0;

    // wait for x,W read
    mchan_barrier(dma_read_evt_x);
    mchan_free(dma_read_evt_x);
    // check if last in any dimension
    last_nof_exec = last_nof_load;
    last_nif_exec = last_nif_load;
    last_h_exec = last_h_load;
    last_w_exec = last_w_load;
    last_nof_load = (_i_nof_load+1 == 1) ? 1 : 0;
    last_nif_load = (_i_nof_load+1 == 1) ? 1 : 0;
    last_h_load = (_i_h_load+1 == 2) ? 1 : 0;
    last_w_load = (_i_w_load+1 == 2) ? 1 : 0;

    // compute double buffering offsets and update db state
    db_x = !db_state_x ? 17408 : 0;
    db_y = !db_state_y ? 4352 : 0;
    exec_db_x = db_state_x ? 17408 : 0;
    db_state_x = ! db_state_x;

    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single filter_out
      db_state_y = ! db_state_y;

    if(iter<1*2*2-1) {
      x_tile_size_nif = (last_nif_load) ? 32 : 32;
      x_tile_size_h   = (last_h_load)   ? 15 : 16;
      x_tile_size_w   = (last_w_load)   ? 21 : 34;
      x_tile_size_byte = x_tile_size_nif*x_tile_size_h*x_tile_size_w*1;
      x_length_h_px = (last_h_load) ? 15 : 16;
      x_length_nif_byte = (last_nif_load)   ? 32 : 32;
      // additionally overlap by padding for the first tile after a border one
      //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      pad_offset_h=0, pad_offset_w=0;
      if(_i_h_load > 0)
        pad_offset_h = 1;
      if(_i_w_load > 0)
        pad_offset_w = 1;

      dory_dma_memcpy_3d_custom(
        dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, 16, 34, 32, 54, 32,  0, 0,0, pad_offset_h, pad_offset_w, 0, 1), // extern
        (l1_buffer + 0) + db_x, // loc
        x_tile_size_byte, // size: dimension of the buffer
        1728, // stride_1: stride for the 3d copy: if we have to copy on n_features axis, this is the stride to change from first 2D space to the next ones.
        32, // stride_0: stride to be passed to 2d_copy: the dimension w of the in image
        x_length_h_px,// length_2: how many 2_d copies we need -> the dimension of the tile in n_features direction
        x_length_nif_byte, // length_0: legnth of the 1_d copy, the length of tile in w direction
        1, // dir
        &dma_read_evt_x // copy
        );
      y_tile_size_h   = (last_h_load)   ? 8 : 8;
      y_tile_size_w   = (last_w_load)   ? 11 : 17;
    }
    x = (char *) (l1_buffer + 0 + exec_db_x);
    y = (char *) (l1_buffer + 34820 + db_y);
    x_tile_size_nif_exec = (last_nif_exec) ? 32 : 32;
    x_tile_size_h_exec   = (last_h_exec)   ? 15 : 16;
    x_tile_size_w_exec   = (last_w_exec)   ? 21 : 34;

    y_tile_size_nof = (last_nof_exec) ? 32 : 32;
    y_tile_size_h   = (last_h_exec)   ? 8 : 8;
    y_tile_size_w   = (last_w_exec)   ? 11 : 17;
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*1;
    y_length_h_px = (last_h_exec) ? 8 : 8;
    y_length_nof_byte = (last_nof_exec)   ? 32 : 32;
    p_r = 0;
    p_l = 0;
    p_t = 0;
    p_b = 0;
    if(_i_h_exec == 0 && _i_w_exec == 0) {
      p_l = 1;
      p_t = 1;
    }
    else if (_i_h_exec == 0) {
      p_t = 1;
    }
    else if (_i_w_exec == 0) {
      p_l = 1;
    }
    if(_i_h_exec == 2-1 && _i_w_exec == 2-1) {
      p_r = 1;
      p_b = 1;
    }
    else if (_i_h_exec == 2-1) {
      p_b = 1;
    }
    else if (_i_w_exec == 2-1) {
      p_r = 1;
    }
  }
  rt_team_barrier();

// aggiungere padding su tutti i lati, acc_out, and filter asymettric
  pulp_nn_maxpool_u8(
    x,
    x_tile_size_w_exec,
    x_tile_size_h_exec,
    x_tile_size_nif_exec,
    2,
    p_t,
    p_b,
    p_l,
    p_r,
    2,
    y_tile_size_w,
    y_tile_size_h,
    im2col,
    y,
    0,
    0,
    flag_first_ch_out
    );
if(rt_core_id()==0){
    // wait for DMA write
      if(iter) {
        mchan_barrier(dma_write_evt_y);
        mchan_free(dma_write_evt_y);
      }
      dory_dma_memcpy_3d_custom(
        dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, 8, 17, 32, 28, 32, 0, 0, 0, 0, 0, 0, 1), // ext
        (l1_buffer + 34820) + db_y, // loc
        y_tile_size_byte, // size
        896, // stride_1
        32, // stride_0
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
