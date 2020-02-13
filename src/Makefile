#------------------------------------------------------------------------------#
# Copyright (C) 2018-2019 ETH Zurich, Switzerland                              #
# All rights reserved.                                                         #
#                                                                              #
# Licensed under the Apache License, Version 2.0 (the "License");              #
# you may not use this file except in compliance with the License.             #
# See LICENSE.apache.md in the top directory for details.                      #
# You may obtain a copy of the License at                                      #
#                                                                              #
#     http://www.apache.org/licenses/LICENSE-2.0                               #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the License is distributed on an "AS IS" BASIS,            #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the License for the specific language governing permissions and          #
# limitations under the License.                                               #
#                                                                              #
# File:    Makefile                                                            #
# Author:  Daniele Palossi <dpalossi@iis.ee.ethz.ch>                           #
# Date:    10.04.2019                                                          #
#------------------------------------------------------------------------------#

# Uncomment 'DEBUG' to enable debugging support (e.g., float prints)
# DEBUG				= true
pulpChip			= GAP
PULP_APP			= PULPDroNet
WEIGHTS_DIR			= $(CURDIR)/../weights
TILER_DIR			= $(CURDIR)/autotiler
FINDNET_INC_DIR		= $(CURDIR)/findnet/inc
PULP_APP_SRCS		= PULPDronet.c PULPDronetKernels.c PULPDronetKernelsInit.c $(TILER_DIR)/src/CNN_BasicKernels.c

# PULP_APP_SRCS += findnet/src/test_template.c # main, not used if DroNet is real main
PULP_APP_SRCS += findnet/src/network.c findnet/src/dory.c findnet/src/hyperram_aligned.c
PULP_APP_SRCS += findnet/src/pulp_nn_conv_i8_u8.c findnet/src/pulp_nn_matmul_4x2_i8_u8.c findnet/src/pulp_nn_relu_u8.c
PULP_APP_SRCS += findnet/src/pulp_nn_dw_conv_i8_u8.c findnet/src/pulp_nn_conv_i8_u8_first.c findnet/src/pulp_nn_quant_u8.c findnet/src/pulp_nn_bn_quant_u8.c 
PULP_APP_SRCS += findnet/src/pulp_nn_avgpool_u8.c findnet/src/pulp_nn_maxpool_u8.c
PULP_APP_SRCS += findnet/src/utils.c
PULP_APP_SRCS += findnet/src/mem_controller.c

PULP_APP_SRCS += findnet/src/layerMaxPool1.c
PULP_APP_SRCS += findnet/src/layerConvBNRelu0.c
PULP_APP_SRCS += findnet/src/layerConvBNRelu5.c
PULP_APP_SRCS += findnet/src/layerConvBNRelu4.c
PULP_APP_SRCS += findnet/src/layerAveragePoolRelu8.c
PULP_APP_SRCS += findnet/src/layerConvBNRelu3.c
PULP_APP_SRCS += findnet/src/layerGemm9.c
PULP_APP_SRCS += findnet/src/layerConvBNRelu6.c
PULP_APP_SRCS += findnet/src/layerConvBNRelu2.c
PULP_APP_SRCS += findnet/src/layerConvBNRelu7.c
PULP_APP_SRCS += ImgIO.c
ifndef CORE
CORE=1
else
CORE= $(CORE)
endif

pulpFc				= 1
RM					= rm -rf
PULP_ARCH_CFLAGS	= -mPE=8 -mFC=1
PULP_ARCH_LDFLAGS	= -mPE=8 -mFC=1
ifdef DEBUG
	PULP_CFLAGS		+= -Os -DDEBUG
else
	PULP_CFLAGS		+= -Os 
endif
PULP_CFLAGS			+= -DNUM_CORES=$(CORE) -w -mno-memcpy -fno-tree-loop-distribute-patterns -fdata-sections -ffunction-sections -Wall -Wno-maybe-uninitialized -Wno-unused-but-set-variable
PULP_CFLAGS 		+= -I$(TILER_DIR)/include -I$(CURDIR) -I$(FINDNET_INC_DIR)
PULP_LDFLAGS		+= -flto -Wl,--gc-sections
USER_CONFIG			= $(CURDIR)/config.ini


all::

autotiler: clean_tiler generate

build_tiler:
	gcc -o AutoTiler -I$(TILER_DIR)/include -I$(CURDIR) PULPDronetGenerator.c $(TILER_DIR)/src/CNN_Generator.c $(TILER_DIR)/lib/libtile.a
	
generate: build_tiler
	./AutoTiler -K -O

clean::
	$(RM) BUILD

clean_tiler: clean
	$(RM) AutoTiler PULPDronetKernels.* PULPDronetKernelsInit.*

.PHONY: generate clean


################################### WEIGHTS ####################################
override CONFIG_OPT += 	flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_1.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_2.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_3.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_4.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_5.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_6.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_7.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_8.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_9.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_conv2d_10.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_dense_1.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/weights_dense_2.hex 


#################################### BIASES ####################################
override CONFIG_OPT +=	flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_1.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_2.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_3.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_4.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_5.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_6.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_7.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_8.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_9.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_conv2d_10.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_dense_1.hex \
						flash/fs/files=$(WEIGHTS_DIR)/binary/bias_dense_2.hex


################################# CAMERA IMGS ##################################

# Use this if you want to test a single image
override CONFIG_OPT += camera/image-stream=$(CURDIR)/../dataset/Himax_Dataset/test_2/frame_22.pgm

# Use this if you want to test the dataset
# override CONFIG_OPT += camera/image-stream=$(EXT_INPUT)


#include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
include $(GAP_SDK_HOME)/tools/rules/pulp_rules.mk