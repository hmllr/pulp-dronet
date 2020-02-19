/*----------------------------------------------------------------------------*
 * Copyright (C) 2018-2019 ETH Zurich, Switzerland                            *
 * All rights reserved.                                                       *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * See LICENSE.apache.md in the top directory for details.                    *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *     http://www.apache.org/licenses/LICENSE-2.0                             *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 *                                                                            *
 * File:    PULPDronet.c                                                      *
 * Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                         *
 * Date:    10.04.2019                                                        *
 *----------------------------------------------------------------------------*/


#include <stdio.h>
#include "Gap8.h"
#include "config.h"
#include "Utils.h"
#include "network.h"
#include "rt/rt_api.h"
#ifdef SHOW_IMAGES
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <fcntl.h>
	#include "ImgIO.h"
#endif


char *			L2_base[NUM_L2_BUFF];
#ifdef PROFILE_CL
static int				perf_exe_cum_cl[NLAYERS];
static int				perf_mem_cum_cl[NLAYERS];
#endif
static rt_camera_t *	camera;
static int				imgTransferDone = 0;
static short int *		L2_image;
static int 				image_size_bytes;
static short int		SPIM_tx[SPIM_BUFFER/2]; // divided by 2 because in bytes
static short int		SPIM_rx[SPIM_BUFFER/2]; // divided by 2 because in bytes


static void enqueue_capture();




static void end_of_frame() {

	rt_cam_control(camera, CMD_PAUSE, NULL);

#if CROPPING == 1
	unsigned char * origin 		= (unsigned char *) L2_image;
	unsigned char * ptr_crop 	= (unsigned char *) L2_image;
	int 			init_offset = CAM_FULLRES_W * LL_Y + LL_X; 
	int 			outid 		= 0;
	
	for(int i=0; i<CAM_CROP_H; i+=DS_RATIO) {	
		rt_event_execute(NULL, 0);
		unsigned char * line = ptr_crop + init_offset + CAM_FULLRES_W * i;
		unsigned char * nextline = ptr_crop + init_offset + CAM_FULLRES_W * (i+1);
		unsigned char * nextnextline = ptr_crop + init_offset + CAM_FULLRES_W * (i+2);
		for(int j=0; j<CAM_CROP_W; j+=DS_RATIO) {
			origin[outid] = (line[j] + line[j+1] + line[j+2] + nextline[j] + nextline[j+1] + nextline[j+2] + nextnextline[j] + nextnextline[j+1] + nextnextline[j+2])/9;
			#ifdef PRINT_IMAGES
				printf("%d,",origin[outid]);
			#endif PRINT_IMAGES
			outid++;
		}
	}
#endif

#ifdef SHOW_IMAGES
  WriteImageToFile("../../../NN_input_image.ppm",CAM_CROP_W/DS_RATIO,CAM_CROP_H/DS_RATIO,L2_image);
#endif //SHOW_IMAGES

	imgTransferDone = 1;
}


static void enqueue_capture() {

	rt_cam_control(camera, CMD_START, NULL);

	rt_camera_capture(camera, (unsigned char*)L2_image, CAM_WIDTH*CAM_HEIGHT*sizeof(unsigned char), rt_event_get(NULL, end_of_frame, NULL));
}



/*----------------.  .----------------.  .----------------.  .-----------------.
| .--------------. || .--------------. || .--------------. || .--------------. |
| | ____    ____ | || |      __      | || |     _____    | || | ____  _____  | |
| ||_   \  /   _|| || |     /  \     | || |    |_   _|   | || ||_   \|_   _| | |
| |  |   \/   |  | || |    / /\ \    | || |      | |     | || |  |   \ | |   | |
| |  | |\  /| |  | || |   / ____ \   | || |      | |     | || |  | |\ \| |   | |
| | _| |_\/_| |_ | || | _/ /    \ \_ | || |     _| |_    | || | _| |_\   |_  | |
| ||_____||_____|| || ||____|  |____|| || |    |_____|   | || ||_____|\____| | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------*/ 

int main() {

#ifdef SHOW_IMAGES
  printf("Connecting to bridge...\n");
  //Open Debug bridge connection for FILE IO
  rt_bridge_connect(1, NULL);
  printf("Connection done.\n");

#endif //SHOW_IMAGES

#ifdef VERBOSE
	printf("FC Launched\n");
#endif
	// set up for H/nH NN
	network_setup();



	// TODO: not sure what we need this for, but won't execute the loop multiple times if missing (was in hyperflash init in dronet)
	if(rt_event_alloc(NULL, 1)) return -1;



/* --------------------------- MEMORY ALLOCATION ---------------------------- */

	
	for(int i=0; i<NUM_L2_BUFF; i++) {
		L2_base[i] = rt_alloc(RT_ALLOC_L2_CL_DATA, L2_buffers_size[i]);
		//L2_next_free[i] = L2_base[i];

#ifdef VERBOSE
		printf("L2 Buffer alloc\t%dB\t@ 0x%08x:\t%s\n", L2_buffers_size[i], (unsigned int)L2_base[i], L2_base[i]?"Ok":"Failed");
#endif
		if(L2_base[i] == NULL) return -1;
	}

	// allocate the memory of L2 for the image buffer
	int image_size_bytes = MAX((CAM_CROP_W/DS_RATIO)*(CAM_CROP_H/DS_RATIO)*sizeof(short int), CAM_FULLRES_W*CAM_FULLRES_H*sizeof(unsigned char));
	L2_image = rt_alloc(RT_ALLOC_L2_CL_DATA, image_size_bytes);
#ifdef VERBOSE
	printf("L2 Image alloc\t%dB\t@ 0x%08x:\t%s\n", image_size_bytes, (unsigned int) L2_image, L2_image?"Ok":"Failed");
#endif
	if(L2_image == NULL) return -1;


/* -------------------------------------------------------------------------- */


/* --------------------------- SPIM CONFIGURATION --------------------------- */

#ifdef SPI_COMM
	// configure the SPI device
	rt_spim_conf_t spim_conf;
	// get default configuration
	rt_spim_conf_init(&spim_conf);
	spim_conf.max_baudrate = 2000000;
	spim_conf.id = 1; 
	spim_conf.cs = 0;
	spim_conf.wordsize = RT_SPIM_WORDSIZE_8;

	// open the device
	rt_spim_t *spim = rt_spim_open(NULL, &spim_conf, NULL);
#ifdef VERBOSE
	printf("SPI Master opening:\t\t\t%s\n", spim?"Ok":"Failed");
#endif
	if(spim == NULL) return -1;

#endif // SPI_COMM

/* -------------------------------------------------------------------------- */


/* -------------------------- CAMERA CONFIGURATION -------------------------- */

	rt_cam_conf_t cam_conf;

	cam_conf.type				= RT_CAM_TYPE_HIMAX;
	cam_conf.resolution 		= QVGA;
	cam_conf.format 			= HIMAX_MONO_COLOR;
	cam_conf.fps 				= fps30;
	cam_conf.slice_en 			= DISABLE;
	cam_conf.shift 				= 0;
	cam_conf.frameDrop_en 		= DISABLE;
	cam_conf.frameDrop_value 	= 0;
	cam_conf.cpiCfg 			= UDMA_CHANNEL_CFG_SIZE_8;
#if PLATFORM==2 // GAPuino
	cam_conf.control_id			= 1;
#else // PULP-Shield or GV-SoC
	cam_conf.control_id			= 0;
#endif
	cam_conf.id					= 0;

	camera = rt_camera_open(NULL, &cam_conf, 0);

#ifdef VERBOSE
		printf("HiMax camera opening:\t\t\t%s\n", camera?"Ok":"Failed");
#endif
	if(camera == NULL) return -1;

	himaxRegWrite(camera, IMG_ORIENTATION,	0x00);	//	Img orientation		[Def: 0x10]
	himaxRegWrite(camera, AE_TARGET_MEAN, 	0x4E);	//	AE target mean 		[Def: 0x3C]
	himaxRegWrite(camera, AE_MIN_MEAN,		0x1C);	//	AE min target mean 	[Def: 0x0A]
	himaxRegWrite(camera, MAX_AGAIN_FULL,	0x02);	//	Max AGAIN full res 	[Def: 0x00]
	himaxRegWrite(camera, MIN_AGAIN, 		0x00);	//	Min AGAIN 			[Def: 0x00]
	himaxRegWrite(camera, BLC_TGT, 			0x20);	//	Black level target 	[Def: 0x20]
	himaxRegWrite(camera, ANALOG_GAIN,		0x00);	//	Analog Global Gain 	[Def: 0x00]
	himaxRegWrite(camera, DIGITAL_GAIN_H, 	0x03);	//	Digital Gain High 	[Def: 0x01]
	himaxRegWrite(camera, DIGITAL_GAIN_L, 	0xFC);	//	Digital Gain Low 	[Def: 0x00]
	himaxRegWrite(camera, MAX_DGAIN,		0xF0);	//	Max DGAIN 			[Def: 0xC0]
	himaxRegWrite(camera, MIN_DGAIN, 		0x60);	//	Min DGAIN 			[Def: 0x40]
	himaxRegWrite(camera, SINGLE_THR_HOT, 	0xFF);	//	single hot px th 	[Def: 0xFF]
	himaxRegWrite(camera, SINGLE_THR_COLD,	0xFF);	//	single cold px th 	[Def: 0xFF]

	rt_cam_control(camera, CMD_INIT, 0);

#if defined(CROPPING) && CROPPING == 0
	rt_img_slice_t slicer;
	slicer.slice_ll.x = LL_X;
	slicer.slice_ll.y = LL_Y;
	slicer.slice_ur.x = UR_X;
	slicer.slice_ur.y = UR_Y;
	
	rt_cam_control(camera, CMD_START, 0);
	rt_cam_control(camera, CMD_SLICE, &slicer);
#endif

	// wait the camera to setup
	if(rt_platform() == ARCHI_PLATFORM_BOARD)
		rt_time_wait_us(1000000);

/* -------------------------------------------------------------------------- */


/* ------------------------- CAMERA 1st ACQUISITION ------------------------- */

	// grab the first frame in advance, because this requires some extra time
	enqueue_capture();
	
	// wait on input image transfer 
	while(imgTransferDone==0) {
		rt_event_yield(NULL);
	}

/* -------------------------------------------------------------------------- */

/* ----------------------------- RUN PULP-DRONET ---------------------------- */

	volatile int iter = 0;

#ifndef DATASET_TEST
	while(1) {
#endif
#ifdef PROFILE_FC
		rt_perf_t perf_fc;
		rt_perf_init(&perf_fc);
		rt_perf_conf(&perf_fc, (1<<RT_PERF_CYCLES));
		rt_perf_reset(&perf_fc);
		rt_perf_start(&perf_fc);
#endif

		// wait on input image transfer 
				while(imgTransferDone==0) {
			rt_event_yield(NULL);
		}
		imgTransferDone=0;


		// execute the H/nH NN on the cluster
	
	  	char head = network_run_FabricController(L2_image);
	  	SPIM_tx[0] = PULP_NAV_MSG_TYPE + (PULP_NAV_MSG_HEAD << 8);
	  	SPIM_tx[PULP_MSG_HEADER_LENGTH/2 + 0] = head << 8; //3rd byte of SPIM_tx

	  	enqueue_capture();
	  	
#ifdef SPI_COMM
		// SPI write out result
		rt_spim_send(spim, SPIM_tx, SPIM_BUFFER*8, RT_SPIM_CS_AUTO, NULL);
#endif

#ifdef PROFILE_FC
		rt_perf_stop(&perf_fc);
		rt_perf_save(&perf_fc);
		printf("FC Cycles:\t\t%d\n", rt_perf_get(&perf_fc, RT_PERF_CYCLES));
#endif

#ifdef VERBOSE
		printf("Result[headprob*255]:\t%d\n", SPIM_tx[PULP_MSG_HEADER_LENGTH/2 + 0]);
#endif

		iter++;
#ifndef DATASET_TEST
	}
#endif

/* -------------------------------------------------------------------------- */


/* --------------------------- FINAL FREE/CLOSE ----------------------------- */

	// close camera module
	rt_camera_close(camera, 0);

#ifdef SPI_COMM
	// close SPI interface
	rt_spim_close(spim, NULL);
#endif

	rt_free(RT_ALLOC_L2_CL_DATA, L2_image, image_size_bytes);

	for(int i=0; i<NUM_L2_BUFF; i++)
		rt_free(RT_ALLOC_L2_CL_DATA, L2_base[i], L2_buffers_size[i]);


	return 0;
}