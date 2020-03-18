#!/bin/bash
# Run this script in the directory to be processed
#
# Inputs:
# $1 - channel, e.g. Ex_642_Em_3
# $X_STEP_SIZE - size of X step in microns
# $Y_VOXEL_SIZE - size of voxel in the Y direction in microns
export channel=$1

if [ -z "$X_STEP_SIZE" ]; then export X_STEP_SIZE=1.28; fi
if [ -z "$Y_VOXEL_SIZE" ]; then export Y_VOXEL_SIZE=1.8; fi

set -x
#
# test for single directory
#
if [ `find $channel -name "*.dcimg" | wc -l` == 1 ]; then
  export SINGLE_CHANNEL=1
  export SINGLE_X=`ls $channel`
  export SINGLE_XY=`ls $channel/"$SINGLE_X"`
else
  export SINGLE_CHANNEL=0
fi
#
# For each x coordinate
#
for x in `ls $channel`;
do
    for xy in `ls $channel/$x`;
    # For the x_y coordinate combinations
    do
	#
	# Make the directories we will need for X and Y
	#
	mkdir -p "$channel"_raw/"$x"/"$xy"
	mkdir -p "$channel"_destriped_precomputed/"$x"/"$xy"
	export dcimg=`ls $channel/$x/$xy/*.dcimg`
	#
	# Convert images from .dcimg format to TIFF
	#
	dcimg2tif\
	    --n-workers 24 \
	    --rotate-90 3 \
	    --flip-ud \
	    --input "$dcimg" \
	    --output-pattern "$channel"_raw/"$x"/"$xy"/img_%05d.tiff
	#
	# Destripe
	#
	pystripe \
	    --input "$channel"_raw/"$x"/"$xy" \
	    --output "$channel"_destriped/"$x"/"$xy" \
	    --sigma1 128 \
	    --sigma2 512 \
	    --wavelet db5 \
	    --crossover 10 \
	    --workers 48
	#
	# Convert the stack of TIFFs to an oblique blockfs volume
	#
	stack2oblique \
	    --n-workers 24 \
	    --n-writers 12 \
	    --input "$channel"_destriped/"$x"/"$xy"/"img*.tiff" \
            --output "$PWD"/"$channel"_destriped_precomputed/"$x"/"$xy" \
	    --levels 4
    done
done
#
# Stitch all of the oblique stacks into a unitary precomputed volume
#
if [ $SINGLE_CHANNEL == 0 ] then
oblique2stitched \
    --input $PWD/"$channel"_destriped_precomputed \
    --output $PWD/"$channel"_destriped_precomputed_stitched \
    --levels 7 \
    --x-step-size "$X_STEP_SIZE" \
    --y-voxel-size "$Y_VOXEL_SIZE" \
    --n-writers 12 \
    --n-workers 24
#
# Convert the precomputed volume's level 1 blockfs to TIFFs
#
blockfs2tif \
    --input "$channel"_destriped_precomputed_stitched/1_1_1/precomputed.blockfs \
    --output-pattern "$channel"_destriped_stitched/img_%04d.tiff
else
  blockfs2tif \
    --input "$channel"_destriped_precomputed/"$SINGLE_X"/"$SINGLE_XY"/1_1_1/precomputed.blockfs \
    --output-pattern "$channel"_destriped_stitched/img_%04d.tiff
fi
#
# Clean up by deleting all intermediate files
#
rm -r "$channel"_raw
rm -r "$channel"_destriped
rm -r "$channel"_destriped_precomputed

