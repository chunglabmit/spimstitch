#!/bin/bash
# Run this script in the directory to be processed
#
# Inputs:
# $1 - channel, e.g. Ex_642_Em_3
# $X_STEP_SIZE - size of X step in microns
# $Y_VOXEL_SIZE - size of voxel in the Y direction in microns
# $Z_OFFSET - the offset in voxels between DCIMG stacks in the Z direction
set -e
export channel=$1

if [ -z "$X_STEP_SIZE" ]; then export X_STEP_SIZE=1.28; fi
if [ -z "$Y_VOXEL_SIZE" ]; then export Y_VOXEL_SIZE=1.8; fi
if [ -z "$Z_OFFSET" ]; then export Z_OFFSET=2048; fi

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
      for dcimg in `ls $channel/$x/$xy/*.dcimg`;
        do
  #
  # Find Z from dcimg
  #
  z=`echo $dcimg | cut -d. -f1`
	#
	# Make the directories we will need for X and Y
	#
	mkdir -p "$channel"_raw/"$x"/"$xy"/"$z"
	mkdir -p "$channel"_destriped_precomputed/"$x"/"$xy"/"$z"
	#
	# Convert images from .dcimg format to TIFF
	#
	dcimg2tif\
	    --n-workers 24 \
	    --rotate-90 3 \
	    --flip-ud \
	    --input "$dcimg" \
	    --output-pattern "$channel"_raw/"$x"/"$xy"/"$z"/img_%05d.tiff
	#
	# Destripe
	#
	pystripe \
	    --input "$channel"_raw/"$x"/"$xy"/"$z" \
	    --output "$channel"_destriped/"$x"/"$xy"/"$z" \
	    --lightsheet \
	    --workers 48
  #	    --sigma1 128 \
  #	    --sigma2 512 \
	#    --wavelet db5 \
	#    --crossover 10 \
	#
	# Convert the stack of TIFFs to an oblique blockfs volume
	#
	if [ $SINGLE_CHANNEL == 0 ]
	then
    stack2oblique \
        --n-workers 24 \
        --n-writers 12 \
        --input "$channel"_destriped/"$x"/"$xy"/"$z"/"img*.tiff" \
        --output "$PWD"/"$channel"_destriped_precomputed/"$x"/"$xy"/"$z" \
        --levels 4
  else
    stack2oblique \
        --n-workers 24 \
        --n-writers 12 \
        --input "$channel"_destriped/"$x"/"$xy"/"$z"/"img*.tiff" \
        --output "$PWD"/"$channel"_destriped_precomputed \
        --levels 5
  fi
	      done
    done
done
#
# Stitch all of the oblique stacks into a unitary precomputed volume
#
if [ $SINGLE_CHANNEL == 0 ]
then
oblique2stitched \
    --input $PWD/"$channel"_destriped_precomputed \
    --output $PWD/"$channel"_destriped_precomputed_stitched \
    --levels 7 \
    --x-step-size "$X_STEP_SIZE" \
    --y-voxel-size "$Y_VOXEL_SIZE" \
    --z-offset "$Z_OFFSET" \
    --n-writers 11 \
    --n-workers 24
#
# Convert the precomputed volume's level 1 blockfs to TIFFs
#
blockfs2tif \
    --input "$channel"_destriped_precomputed_stitched/1_1_1/precomputed.blockfs \
    --output-pattern "$channel"_destriped_stitched/img_%04d.tiff
else
  blockfs2tif \
    --input "$channel"_destriped_precomputed/1_1_1/precomputed.blockfs \
    --output-pattern "$channel"_destriped_stitched/img_%04d.tiff
fi
#
# Clean up by deleting all intermediate files
#
rm -r "$channel"_raw
rm -r "$channel"_destriped
if [ $SINGLE_CHANNEL == 0 ]
then
  rm -r "$channel"_destriped_precomputed
fi

