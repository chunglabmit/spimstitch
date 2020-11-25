#!/bin/bash
#
# Run this script in the directory to be processed.
#
# Inputs:
#   $Y_VOXEL_SIZE - size of a voxel in microns in the camera field
#   $X_STEP_SIZE - size of an X step
#
export channel=$1
if [ -z "$X_STEP_SIZE" ]; then export X_STEP_SIZE=1.28; fi
if [ -z "$Y_VOXEL_SIZE" ]; then export Y_VOXEL_SIZE=1.8; fi

set -x
#
# test for single directory
#
if [ `find $channel -name "*.dcimg" | wc -l` == 1 ]; then
  dcimg_path=`find $channel -name "*.dcimg"`
  dcimg2tif --n-workers 24 \
            --rotate-90 3 \
            --flip-ud \
            --input dcimg_path \
            --output-pattern "$channel"_raw/img_%05d.tiff
  pystripe --input "$channel"_raw \
           --lightsheet \
           --workers 40
  precomputed-tif \
      --source "$channel"_destriped/"*.tiff" \
      --dest "$channel"_precomputed \
      --levels 5 \
      --format blockfs \
      --n-cores 24
else
  for x in `ls $channel`;
do
    for xy in `ls $channel/$x`;
    # For the x_y coordinate combinations
    do
      for dcimg_path in `ls $channel/$x/$xy/*.dcimg`;
        do
  #
  # Cut away the directory
  #
  dcimg=`basename $dcimg_path`
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
	    --input "$dcimg_path" \
	    --output-pattern "$channel"_raw/"$x"/"$xy"/"$z"/img_%05d.tiff
	#
	# Destripe
	#
	pystripe \
	    --input "$channel"_raw/"$x"/"$xy"/"$z" \
	    --output "$channel"_destriped/"$x"/"$xy"/"$z" \
	    --lightsheet \
	    --workers 40
	precomputed-tif \
	   --source "$channel"_destriped/"$x"/"$xy"/"$z"/"*.tiff" \
	   --dest "$channel"_destriped_precomputed/"$x"/"$xy"/"$z" \
	   --format blockfs \
	   --levels 1 \
	   --n-cores 24
	    done
	  done
	done
	nonoblique2stitched \
	  --input "$channel"_destriped_precomputed \
	  --output "$channel"_destriped_stitched_precomputed \
	  --voxel-size $Y_VOXEL_SIZE \
	  --x-step-size $X_STEP_SIZE

fi
