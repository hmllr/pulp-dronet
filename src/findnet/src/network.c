#include "hyperram_aligned.h"
#include "mem_controller.h"
#include "network.h"
#include "pulp.h"
#include "dory.h"
#include "rt/rt_api.h"
#include "utils.h"
#include "kernels.h"
#include "layer_init.h"
#include "layerConvBNRelu3.h"
#include "layerConvBNRelu5.h"
#include "layerConvBNRelu2.h"
#include "layerAveragePoolRelu8.h"
#include "layerGemm9.h"
#include "layerConvBNRelu0.h"
#include "layerConvBNRelu4.h"
#include "layerConvBNRelu7.h"
#include "layerConvBNRelu6.h"
#include "layerMaxPool1.h"
#include "PULPDronetKernels.h"

#define FLASH_BUFF_SIZE 128
//#define VERBOSE 1
//#define VERBOSE_PERFORMANCE
#define VERBOSE_RESULT
//#define TEST_IMAGE
static const char * L3_weights_files[] = {
  "ConvBNRelu0_weights.hex", "ConvBNRelu2_weights.hex", "ConvBNRelu3_weights.hex", "ConvBNRelu4_weights.hex", "ConvBNRelu5_weights.hex", "ConvBNRelu6_weights.hex", "ConvBNRelu7_weights.hex", "Gemm9_weights.hex"
};
extern char *     L2_base[1];//NUM_L2_BUFF/2];
const int activations_size[] = {
  6480, 51840, 14336, 14336, 14336, 7168, 7168, 3584, 3584, 128, 1
};
static int L3_weights_size[8];
static int L3_weights;
static int activations_input;
extern rt_hyperram_t* hyperram;



#ifdef VERBOSE
static void check_layer(char *output, int check_sum_true, int dim) {
  int checksum = 0;
  char *ptr = (char *) output;
  for(int j=0; j<dim; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum in/out Layer :\tOk dim:%d\n", dim);
  else 
    printf("Checksum in/out Layer :\tFailed [%u vs. %u] dim: %d\n", checksum, check_sum_true, dim);
}


static void check_layer_weight(char *weight, int check_sum_true, int dim) {
  int checksum = 0;
  char *ptr = (char *) weight;
  for(int j=0; j<dim; j++) {
    checksum += ptr[j];
  }

  if(check_sum_true == checksum)
    printf("Checksum weight/bias Layer :\tOk\n");
  else 
    printf("Checksum weight/bias Layer :\tFailed [%u vs. %u]\n", checksum, check_sum_true);
}
#endif 

/* Moves the weights and the biases from hyperflash to hyperram */
int network_setup()
{
  /* PADFRAME CONFIGURATION */
  rt_padframe_profile_t *profile_hyper = rt_pad_profile_get("hyper");
  rt_padframe_set(profile_hyper);

  /* FILE-SYSTEM CONFIGURATION*/
  rt_fs_conf_t conf;
  rt_fs_conf_init(&conf);
  rt_fs_t *fs = rt_fs_mount("hyperflash", &conf, NULL);
  /* INITIALIZE HYPER RAM CONFIGURATION */
  rt_hyperram_conf_t hyperram_conf;
  rt_hyperram_conf_init(&hyperram_conf);
  hyperram = rt_hyperram_open("hyperram", &hyperram_conf, NULL);
  if (hyperram == NULL) return -1;

  /* LOAD FILE WITH WEIGHTS FROM HYPERFLASH */
  rt_file_t *file;
  L3_weights = (char *) rt_hyperram_alloc(hyperram, 5000000);
  unsigned int rdDone = 0;
  for (int i=0;i<8;i++)
  {
    file = rt_fs_open(fs, L3_weights_files[i], 0, NULL);
    L3_weights_size[i] = file->size + rdDone;
    int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
    void *flashBuffer = rt_alloc(RT_ALLOC_PERIPH, flashBuffSize);
    // loop on chunk in file
    while(rdDone < (L3_weights_size[i] / sizeof(char))) 
    { 
      // read from HyperFlash
      int size = rt_fs_read(file, flashBuffer, flashBuffSize, NULL);
      // write to HyperRam
      rt_hyperram_write(hyperram, flashBuffer, L3_weights+rdDone, size, NULL);
      rdDone += size / sizeof(char);
    }
    rt_free(RT_ALLOC_PERIPH,flashBuffer, flashBuffSize);
  }
  file = rt_fs_open(fs, "inputs.hex", 0, NULL);
  if (file == NULL)
  {
    printf("file open failed\n");
    return -1;
  }
    activations_input = L3_weights+rdDone;
    rdDone = 0;
    int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
    void *flashBuffer = rt_alloc(RT_ALLOC_PERIPH, flashBuffSize);
    // loop on chunk in file
    while(rdDone < (6480 / sizeof(char))) 
    { 
      // read from HyperFlash
      int size = rt_fs_read(file, flashBuffer, flashBuffSize, NULL);
      // write to HyperRam
      rt_hyperram_write(hyperram, flashBuffer, activations_input+rdDone, size, NULL);
      rdDone += size / sizeof(char);
    }
    rt_free(RT_ALLOC_PERIPH,flashBuffer, flashBuffSize);
  return 1;
}

// on cluster
void cluster_main(void *arg) {
  int *real_arg = (int *) arg;
  network_run(
    (unsigned int) real_arg[0],
    (unsigned int) real_arg[1],
    (short int *)  real_arg[2]
    );
}

void pulp_parallel(void *arg)
{
  rt_team_fork(NUM_CORES, (void *)cluster_main, arg);
}

  int memId;
  char* L2_output;
  char* L2_input;
  char* L2_weights_1;
  char* L2_weights_2;
  char* L2_buffer_allocation;
  int L2_buffer_allocation_end;

   char *l1_buffer;

char network_run_FabricController(short int *   L2_image, void* stacks)
{
  int arg[3];
  arg[0] = (unsigned int) L3_weights_size;
  arg[1] = (unsigned int) hyperram;
  arg[2] = L2_image;

  rt_cluster_call(NULL, 0, pulp_parallel, arg, stacks,1024+1024, 1024+1024, rt_nb_pe(), NULL);

  return *(L2_output);
}

void network_run( 
  unsigned int L3_weights_size,
  unsigned int hyperram,
  short int *   L2_image
  )
{   

  if (rt_core_id()==0)
  {
    L2_buffer_allocation = L2_base[0];//rt_alloc(RT_ALLOC_L2_CL_DATA, 400000);
    L2_buffer_allocation_end = L2_buffer_allocation + 400000;
    l1_buffer = PULP_Dronet_L1_Memory; //rt_alloc(RT_ALLOC_CL_DATA,44000 );
#ifdef VERBOSE
    printf("L2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)L2_buffer_allocation, L2_buffer_allocation?"Ok":"Failed");
#endif
  }

    int d_buffering_weights_t = 0;
    int error_presence = 0;
    int d_buffering_weights_e = 0;
    int d_buffering_inputs = 0;
    int d_buffering_outputs = 0;
    int begin_end_n = 1;
#ifdef VERBOSE
  int check;
#endif
    rt_hyperram_req_t wait_L3_w,wait_L3_b;
    int * L3_weights_size_internal = (int *) L3_weights_size;
    int L3_weights_internal = L3_weights;
    char* exec_weights,*transfer_weights;

  transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
  exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;

  if(rt_core_id()==0)
  {
        dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_input,
            activations_size[0],
            begin_end_n // begin is 1, end is 0
            );
    rt_hyperram_cluster_read_mine(hyperram, L2_input, activations_input, 6480, &wait_L3_w);
    rt_hyperram_cluster_wait(&wait_L3_w);

    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_weights_1,
            L3_weights_size_internal[0],
            begin_end_n // begin is 1, end is 0
            );
    begin_end_n = !begin_end_n;
    transfer_weights = L2_weights_1;
    exec_weights = L2_weights_1;  

    rt_hyperram_cluster_read_mine(hyperram, transfer_weights, L3_weights_internal, L3_weights_size_internal[0], &wait_L3_w);
    rt_hyperram_cluster_wait(&wait_L3_w);

    /* for all layers in a list, instantiate the layer.
    Instantiate the L3 copies again in a double-buffering fashion*/
    /* Instantiate the L3 first memory passage */
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[1],
            begin_end_n // begin is 1, end is 0
            );
    begin_end_n = !begin_end_n;
  }

    rt_perf_t perf2;
    rt_perf_init(&perf2);                       
    rt_perf_conf(&perf2, (1<<RT_PERF_CYCLES));          
    rt_perf_reset(&perf2);                      
    rt_perf_stop(&perf2);                       
    rt_perf_start(&perf2); 
  if(rt_core_id()==0)
  {
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 126682;
    check_layer_weight(exec_weights, check, L3_weights_size_internal[0]) ;
    check = 464548;
    check_layer(L2_input, check, activations_size[0]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerConvBNRelu0(
#ifdef TEST_IMAGE
      L2_input,
#else
      L2_image,
#endif
      L2_output,
      exec_weights,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE
    printf("Layer %d ended: \n", 0);  
    check = 45013;
    check_layer(L2_output, check, activations_size[1]) ;
#endif 
  }
  if(rt_core_id()==0)
  {
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      L3_weights_size_internal[0],
      begin_end_n // begin is 1, end is 0
      );
    //begin_end_n != begin_end_n;

    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      activations_size[0],
      begin_end_n // begin is 1, end is 0
      );
    L2_input = L2_output;
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[2],
            begin_end_n // begin is 1, end is 0
            );

    if (d_buffering_weights_e==1)
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_1,
              L3_weights_size_internal[1]-L3_weights_size_internal[0],
              begin_end_n // begin is 1, end is 0
              );
    }
    else
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_2,
              L3_weights_size_internal[1]-L3_weights_size_internal[0],
              begin_end_n // begin is 1, end is 0
              );
    }  
    d_buffering_weights_t = !d_buffering_weights_t;
    transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
    begin_end_n = !begin_end_n;
        //switching output and input.
  }
  if(rt_core_id()==0)
  {
    rt_hyperram_cluster_read_mine(hyperram, transfer_weights, L3_weights_internal + L3_weights_size_internal[0], L3_weights_size_internal[1]-L3_weights_size_internal[0], &wait_L3_w);
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 45013;
    check_layer(L2_input, check, activations_size[1]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerMaxPool1(
      L2_input,
      L2_output,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE
    printf("Layer %d ended: \n", 1);  
    check = 22753;
    check_layer(L2_output, check, activations_size[2]) ;
#endif 
  }
  if(rt_core_id()==0)
  {
    d_buffering_weights_e = !d_buffering_weights_e;
    exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      activations_size[1],
      begin_end_n // begin is 1, end is 0
      );
    L2_input = L2_output;
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[3],
            begin_end_n // begin is 1, end is 0
            );

    if (d_buffering_weights_e==1)
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_1,
              L3_weights_size_internal[2]-L3_weights_size_internal[1],
              begin_end_n // begin is 1, end is 0
              );
    }
    else
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_2,
              L3_weights_size_internal[2]-L3_weights_size_internal[1],
              begin_end_n // begin is 1, end is 0
              );
    }  
    d_buffering_weights_t = !d_buffering_weights_t;
    transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
    begin_end_n = !begin_end_n;
        //switching output and input.
  }
  if(rt_core_id()==0)
  {
    rt_hyperram_cluster_read_mine(hyperram, transfer_weights, L3_weights_internal + L3_weights_size_internal[1], L3_weights_size_internal[2]-L3_weights_size_internal[1], &wait_L3_w);
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 1232410;
    check_layer_weight(exec_weights, check, L3_weights_size_internal[1] - L3_weights_size_internal[0]) ;
    check = 22753;
    check_layer(L2_input, check, activations_size[2]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerConvBNRelu2(
      L2_input,
      L2_output,
      exec_weights,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE
    printf("Layer %d ended: \n", 2);  
    check = 29309;
    check_layer(L2_output, check, activations_size[3]) ;
#endif 
  }
  if(rt_core_id()==0)
  {
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
              L3_weights_size_internal[1]-L3_weights_size_internal[0],
      begin_end_n // begin is 1, end is 0
      );
    //begin_end_n != begin_end_n;
    rt_hyperram_cluster_wait(&wait_L3_w); 

    d_buffering_weights_e = !d_buffering_weights_e;
    exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      activations_size[2],
      begin_end_n // begin is 1, end is 0
      );
    L2_input = L2_output;
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[4],
            begin_end_n // begin is 1, end is 0
            );

    if (d_buffering_weights_e==1)
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_1,
              L3_weights_size_internal[3]-L3_weights_size_internal[2],
              begin_end_n // begin is 1, end is 0
              );
    }
    else
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_2,
              L3_weights_size_internal[3]-L3_weights_size_internal[2],
              begin_end_n // begin is 1, end is 0
              );
    }  
    d_buffering_weights_t = !d_buffering_weights_t;
    transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
    begin_end_n = !begin_end_n;
        //switching output and input.
  }
  if(rt_core_id()==0)
  {
    rt_hyperram_cluster_read_mine(hyperram, transfer_weights, L3_weights_internal + L3_weights_size_internal[2], L3_weights_size_internal[3]-L3_weights_size_internal[2], &wait_L3_w);
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 1262679;
    check_layer_weight(exec_weights, check, L3_weights_size_internal[2] - L3_weights_size_internal[1]) ;
    check = 29309;
    check_layer(L2_input, check, activations_size[3]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerConvBNRelu3(
      L2_input,
      L2_output,
      exec_weights,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE
    printf("Layer %d ended: \n", 3);  
    check = 22201;
    check_layer(L2_output, check, activations_size[4]) ;
#endif 
  }
  if(rt_core_id()==0)
  {
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
              L3_weights_size_internal[2]-L3_weights_size_internal[1],
      begin_end_n // begin is 1, end is 0
      );
    //begin_end_n != begin_end_n;
    rt_hyperram_cluster_wait(&wait_L3_w); 

    d_buffering_weights_e = !d_buffering_weights_e;
    exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      activations_size[3],
      begin_end_n // begin is 1, end is 0
      );
    L2_input = L2_output;
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[5],
            begin_end_n // begin is 1, end is 0
            );

    if (d_buffering_weights_e==1)
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_1,
              L3_weights_size_internal[4]-L3_weights_size_internal[3],
              begin_end_n // begin is 1, end is 0
              );
    }
    else
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_2,
              L3_weights_size_internal[4]-L3_weights_size_internal[3],
              begin_end_n // begin is 1, end is 0
              );
    }  
    d_buffering_weights_t = !d_buffering_weights_t;
    transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
    begin_end_n = !begin_end_n;
        //switching output and input.
  }
  if(rt_core_id()==0)
  {
    rt_hyperram_cluster_read_mine(hyperram, transfer_weights, L3_weights_internal + L3_weights_size_internal[3], L3_weights_size_internal[4]-L3_weights_size_internal[3], &wait_L3_w);
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 2515182;
    check_layer_weight(exec_weights, check, L3_weights_size_internal[3] - L3_weights_size_internal[2]) ;
    check = 22201;
    check_layer(L2_input, check, activations_size[4]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerConvBNRelu4(
      L2_input,
      L2_output,
      exec_weights,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE
    printf("Layer %d ended: \n", 4);  
    check = 16115;
    check_layer(L2_output, check, activations_size[5]) ;
#endif 
  }
  if(rt_core_id()==0)
  {
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
              L3_weights_size_internal[3]-L3_weights_size_internal[2],
      begin_end_n // begin is 1, end is 0
      );
    //begin_end_n != begin_end_n;
    rt_hyperram_cluster_wait(&wait_L3_w); 

    d_buffering_weights_e = !d_buffering_weights_e;
    exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      activations_size[4],
      begin_end_n // begin is 1, end is 0
      );
    L2_input = L2_output;
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[6],
            begin_end_n // begin is 1, end is 0
            );

    if (d_buffering_weights_e==1)
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_1,
              L3_weights_size_internal[5]-L3_weights_size_internal[4],
              begin_end_n // begin is 1, end is 0
              );
    }
    else
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_2,
              L3_weights_size_internal[5]-L3_weights_size_internal[4],
              begin_end_n // begin is 1, end is 0
              );
    }  
    d_buffering_weights_t = !d_buffering_weights_t;
    transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
    begin_end_n = !begin_end_n;
        //switching output and input.
  }
  if(rt_core_id()==0)
  {
    rt_hyperram_cluster_read_mine(hyperram, transfer_weights, L3_weights_internal + L3_weights_size_internal[4], L3_weights_size_internal[5]-L3_weights_size_internal[4], &wait_L3_w);
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 5006167;
    check_layer_weight(exec_weights, check, L3_weights_size_internal[4] - L3_weights_size_internal[3]) ;
    check = 16115;
    check_layer(L2_input, check, activations_size[5]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerConvBNRelu5(
      L2_input,
      L2_output,
      exec_weights,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE
    printf("Layer %d ended: \n", 5);  
    check = 9996;
    check_layer(L2_output, check, activations_size[6]) ;
#endif 
  }
  if(rt_core_id()==0)
  {
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
              L3_weights_size_internal[4]-L3_weights_size_internal[3],
      begin_end_n // begin is 1, end is 0
      );
    //begin_end_n != begin_end_n;
    rt_hyperram_cluster_wait(&wait_L3_w); 

    d_buffering_weights_e = !d_buffering_weights_e;
    exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      activations_size[5],
      begin_end_n // begin is 1, end is 0
      );
    L2_input = L2_output;
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[7],
            begin_end_n // begin is 1, end is 0
            );

    if (d_buffering_weights_e==1)
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_1,
              L3_weights_size_internal[6]-L3_weights_size_internal[5],
              begin_end_n // begin is 1, end is 0
              );
    }
    else
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_2,
              L3_weights_size_internal[6]-L3_weights_size_internal[5],
              begin_end_n // begin is 1, end is 0
              );
    }  
    d_buffering_weights_t = !d_buffering_weights_t;
    transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
    begin_end_n = !begin_end_n;
        //switching output and input.
  }
  if(rt_core_id()==0)
  {
    rt_hyperram_cluster_read_mine(hyperram, transfer_weights, L3_weights_internal + L3_weights_size_internal[5], L3_weights_size_internal[6]-L3_weights_size_internal[5], &wait_L3_w);
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 10366530;
    check_layer_weight(exec_weights, check, L3_weights_size_internal[5] - L3_weights_size_internal[4]) ;
    check = 9996;
    check_layer(L2_input, check, activations_size[6]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerConvBNRelu6(
      L2_input,
      L2_output,
      exec_weights,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE
    printf("Layer %d ended: \n", 6);  
    check = 2731;
    check_layer(L2_output, check, activations_size[7]) ;
#endif 
  }
  if(rt_core_id()==0)
  {
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
              L3_weights_size_internal[5]-L3_weights_size_internal[4],
      begin_end_n // begin is 1, end is 0
      );
    //begin_end_n != begin_end_n;
    rt_hyperram_cluster_wait(&wait_L3_w); 

    d_buffering_weights_e = !d_buffering_weights_e;
    exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      activations_size[6],
      begin_end_n // begin is 1, end is 0
      );
    L2_input = L2_output;
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[8],
            begin_end_n // begin is 1, end is 0
            );

    begin_end_n = !begin_end_n;
        //switching output and input.
  }
  if(rt_core_id()==0)
  {
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 18506398;
    check_layer_weight(exec_weights, check, L3_weights_size_internal[6] - L3_weights_size_internal[5]) ;
    check = 2731;
    check_layer(L2_input, check, activations_size[7]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerConvBNRelu7(
      L2_input,
      L2_output,
      exec_weights,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE
    printf("Layer %d ended: \n", 7);  
    check = 4124;
    check_layer(L2_output, check, activations_size[8]) ;
#endif 
  }
  if(rt_core_id()==0)
  {
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
              L3_weights_size_internal[6]-L3_weights_size_internal[5],
      begin_end_n // begin is 1, end is 0
      );
    //begin_end_n != begin_end_n;

    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      activations_size[7],
      begin_end_n // begin is 1, end is 0
      );
    L2_input = L2_output;
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[9],
            begin_end_n // begin is 1, end is 0
            );

    if (d_buffering_weights_e==1)
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_1,
              L3_weights_size_internal[7]-L3_weights_size_internal[6],
              begin_end_n // begin is 1, end is 0
              );
    }
    else
    {
      dory_L2_alloc(&L2_buffer_allocation,
              &L2_buffer_allocation_end,
              &L2_weights_2,
              L3_weights_size_internal[7]-L3_weights_size_internal[6],
              begin_end_n // begin is 1, end is 0
              );
    }  
    d_buffering_weights_t = !d_buffering_weights_t;
    transfer_weights = d_buffering_weights_t ? L2_weights_2 : L2_weights_1;
    begin_end_n = !begin_end_n;
        //switching output and input.
  }
  if(rt_core_id()==0)
  {
    rt_hyperram_cluster_read_mine(hyperram, transfer_weights, L3_weights_internal + L3_weights_size_internal[6], L3_weights_size_internal[7]-L3_weights_size_internal[6], &wait_L3_w);
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 4124;
    check_layer(L2_input, check, activations_size[8]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerAveragePoolRelu8(
      L2_input,
      L2_output,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE
    printf("Layer %d ended: \n", 8);  
    check = 109;
    check_layer(L2_output, check, activations_size[9]) ;
#endif 
  }
  if(rt_core_id()==0)
  {
    d_buffering_weights_e = !d_buffering_weights_e;
    exec_weights = d_buffering_weights_e ? L2_weights_2 : L2_weights_1;
    dory_L2_free(&L2_buffer_allocation,
      &L2_buffer_allocation_end,
      activations_size[8],
      begin_end_n // begin is 1, end is 0
      );
    L2_input = L2_output;
    dory_L2_alloc(&L2_buffer_allocation,
            &L2_buffer_allocation_end,
            &L2_output,
            activations_size[10],
            begin_end_n // begin is 1, end is 0
            );

    begin_end_n = !begin_end_n;
        //switching output and input.
  }
  if(rt_core_id()==0)
  {
  }

#ifdef VERBOSE
  if(rt_core_id()==0)
  {
    check = 17073;
    check_layer_weight(exec_weights, check, L3_weights_size_internal[7] - L3_weights_size_internal[6]) ;
    check = 109;
    check_layer(L2_input, check, activations_size[9]) ;
  printf("L2 input %d, L2 output %d, weights %d\n", L2_input, L2_output, exec_weights);
  printf("L1 buffer %d\n", l1_buffer);
  }
#endif  
  rt_team_barrier();
  layerGemm9(
      L2_input,
      L2_output,
      exec_weights,
      l1_buffer
      );
  rt_team_barrier();


//% if i==0:
//  if(rt_core_id()==0)
//  	L2_output[3*32*64+6*32+28] +=1;
//% endif

  if(rt_core_id()==0)
  {
#ifdef VERBOSE_RESULT
    printf("class: %d\n", *(L2_output));
#endif
#ifdef VERBOSE
    printf("Layer %d ended: \n", 9);  
    check = 193;
    int max = -100000;
    int index_max = -1;
    printf("class: %d\n", *(L2_output));
    for(int class = 0; class < activations_size[10]; class ++)
    {
      if (*(L2_output + class) > max) 
      {
        index_max = class;
        max = *(L2_output + class);
      }
    }
    printf("Output class is: %d, Correct Class is %d\n", index_max, 0);
#endif 
  }

rt_perf_stop(&perf2);          
rt_perf_save(&perf2);          
int cid = rt_core_id();   
int perf_cyc =  rt_perf_get(&perf2, RT_PERF_CYCLES) ; 
int MACs = 186400768;
float perf_MAC =  (float)MACs/perf_cyc;
#ifdef VERBOSE_PERFORMANCE
if (cid == 0){
printf("[%d] : num_cycles: %d\n",cid,perf_cyc); 
printf("[%d] : MACs: %d\n",cid,MACs ); 
printf("[%d] : MAC/cycle: %f\n",cid,perf_MAC ); 
printf("[%d] : n. of Cores: %d\n",cid,NUM_CORES); 
}
#endif //VERBOSE_PERFORMANCE
}


