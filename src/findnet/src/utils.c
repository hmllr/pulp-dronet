#include "rt/rt_api.h"
#include "utils.h"
#include "kernels.h"
#include "hyperram_aligned.h"
#include "mchan_test.h"

#define max4(a,b)  		    __builtin_pulp_maxu4(a,b)
#define avg4(a,b)         __builtin_pulp_avg4(a,b)


void pulp_nn_compare_and_replace_if_larger_int8(uint8_t * base,
						                                    uint8_t * target,
						                                    uint16_t length)
{
  uint8_t *pIn = base;
  uint8_t *pCom = target;
  v4u inp;
  v4u com;
  uint16_t cnt = length >> 2;

  while(cnt > 0u)
  {
    inp = *((v4u*)pIn);
    com = *((v4u*)pCom);
    pCom+=4;

    *((v4u*)pIn) = max4(inp, com);
    pIn+=4;
    cnt--;
  }

  uint16_t left = length & 0x3;
  while (left>0u)
  {
    if(*pIn<*pCom)
      *pIn=*pCom;
    pIn++;
    pCom++;
    left--;
  }
}


void pulp_nn_avg_and_replace_int8(int8_t * base,
                                  int8_t * target,
                                  uint16_t length)
{
  int8_t *pIn = base;
  int8_t *pCom = target;
  v4s inp;
  v4s com;
  uint16_t cnt = length >> 2;

  while(cnt > 0u)
  {
    inp = *((v4s*)pIn);
    com = *((v4s*)pCom);
    pCom+=4;

    *((v4s*)pIn) = avg4(inp, com);
    pIn+=4;
    cnt--;
  }
}


void pulp_zero_mem(uint8_t * pBuffer, int size)
{
  v4u* pDst = (v4u *)pBuffer;
  int lfover = size &0x3;
    for (int i=0; i<(size>>2); i++)
    {
      *((v4u*) pBuffer) = (v4u){0,0,0,0};
        pBuffer+=4;
    }
    while(lfover)
    {
      *pBuffer++=0;
      lfover--;
    }
}

void pulp_nn_im2col_int8_dmafree(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 2u;
  unsigned int lfover = blockSize & 0x3;

  for (unsigned int i = 0; i<blkCnt; i++)
  {
    *((v4s*)pOutput) = *((v4s*) pInput);
    pInput+=4;
    pOutput+=4;
  }
  while(lfover)
  {
    *((uint8_t*)pOutput) = *((uint8_t*)pInput);
    pOutput++;
    pInput++;
    lfover--;
  }
}


void pulp_nn_im2col_int8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  mchan_transfer(blockSize, 1, 1, 0, 1, 0, 0, (unsigned int) pInput, (unsigned int) pOutput, 0, 0);
}
