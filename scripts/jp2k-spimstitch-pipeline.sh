#!/bin/bash
# This script compresses the .dcimg files as JPEG2000 and then proceeds using JPEG2000 as the destination type
# Run this script in the directory to be processed
#
# Inputs:
# $1 - channel, e.g. Ex_642_Em_3
# $X_STEP_SIZE - size of X step in microns
# $Y_VOXEL_SIZE - size of voxel in the Y direction in microns
# $Z_OFFSET - the offset in voxels between DCIMG stacks in the Z direction
# $BACKGROUND - the maximum intensity of background pixels
# $PSNR - the signal-to-noise ratio for compression. Set to 0 for lossless compression
# $USE_WAVELETS - use wavelets for destriping, not lightsheet
# $ILLUM_CORR - the illumination correction file to use. If none, one will be computed
# $ALIGN_FILE - the file to use to align scan runs. If none, one will be computed.
#               Subsequent channels should use the first channel's alignment file.
#
set -e
export channel=$1

if [ -z "$X_STEP_SIZE" ]; then export X_STEP_SIZE=1.28; fi
if [ -z "$Y_VOXEL_SIZE" ]; then export Y_VOXEL_SIZE=1.8; fi
if [ -z "$Z_OFFSET" ]; then export Z_OFFSET=2048; fi
if [ -z "$BACKGROUND" ]; then export BACKGROUND=100; fi
if [ -z "$PSNR" ]; then export PSNR=80; fi
if [ -z "$WORKERS" ]; then export WORKERS=48; fi
if [ -z "$ILLUM_CORR" ]; then
  ILLUM_CORR="$channel"-illuc.tiff
  oblique-illum-corr \
    --output $ILLUM_CORR \
    --n-frames 5000 \
    --background $BACKGROUND \
    --n-bins 1024 \
    --values-per-bin 4 \
    --rotate-90 3 \
    --flip-ud \
    `find $channel -name "*.dcimg"`
fi
PYSTRIPE_EXTRA_ARGS="--flat $ILLUM_CORR --dark $BACKGROUND"

if [ -z "$USE_WAVELETS" ]; then
  PYSTRIPE_EXTRA_ARGS+=" --destripe-method lightsheet"
else
  PYSTRIPE_EXTRA_ARGS+=" --destripe-method wavelet --sigma1 128 --sigma2 512 --wavelet db5 --crossover 10"
fi

set -x
ALL_DCIMGS=`find $channel -name "*.dcimg"`
if [ `echo ALL_DCIMGS | wc -w` != "0" ]; then
  export DCIMG2JPEG2000=1
else
  export DCIMG2JPEG2000=0
  ALL_DCIMGS=`find $channel -type d -wholename "*/*/*/*"`
fi
#
# test for single directory
#
if [ `echo $ALL_DCIMGS` == "0" ]; then
  export SINGLE_CHANNEL=1
  export SINGLE_X=`ls $channel`
  export SINGLE_XY=`ls $channel/"$SINGLE_X"`
else
  export SINGLE_CHANNEL=0
fi
#
# For each x coordinate
#
for dcimg_path in $ALL_DCIMGS;
do
  #
  # Cut away the directory
  #
  dcimg=`basename $dcimg_path`
  rest=`dirname $dcimg_path`
  xy=`basename $rest`
  rest=`dirname $rest`
  x=`basename $rest`
  #
  # Find Z from dcimg
  #
  if [ $DCIMG2JPEG2000 == 1 ]; then
    z=`echo $dcimg | cut -d. -f1`
    #
    # Make the directories we will need for X and Y
    #
    jp2k_src="$PWD"/"$channel"_jp2k/"$x"/"$xy"/"$z"
    mkdir -p $jp2k_src
  else
    z=dcimg
    jp2k_src="$PWD"/"$channel"
  fi
	if [ $SINGLE_CHANNEL == 0 ]; then
  	destriped_precomputed="$channel"_destriped_precomputed/"$x"/"$xy"/"$z"
	  intermediate_levels=1
	else
	  intermediate_levels=5
	  destriped_precomputed="$channel"_destriped_precomputed_stitched
	fi
	mkdir -p $destriped_precomputed
	#
	# Convert images from .dcimg to JPEG2000
	#
	if [ $DCIMG2JPEG2000 == 1 ]; then
    dcimg2jp2 \
      --input $dcimg_path \
      --output-pattern $jp2k_src/img_%05d.jp2 \
      --n-workers $WORKERS \
      --psnr $PSNR
  fi
	#
	# Convert images from .dcimg format to oblique precomputed
	#
  dcimg2oblique \
      --n-writers 11 \
      --n-workers $WORKERS \
	    --rotate-90 3 \
	    --flip-ud \
	    --input "$jp2k_src"/"img_*.jp2" \
	    --output "$destriped_precomputed" \
	    $PYSTRIPE_EXTRA_ARGS \
	    --levels $intermediate_levels \
	    --jp2k
done
#
# Stitch all of the oblique stacks into a unitary precomputed volume
#
if [ $SINGLE_CHANNEL == 0 ]; then
  if [ -z $ALIGN_FILE ]; then
    export ALIGN_FILE=$PWD/"$channel"-align.json
    oblique-align \
      --input $PWD/"$channel"_destriped_precomputed \
      --output $ALIGN_FILE \
      --voxel-size $Y_VOXEL_SIZE \
      --x-step-size $X_STEP_SIZE \
      --is-oblique \
      --n-cores $WORKERS \
      --sigma 10 \
      --sample-count 100 \
      --window-size 51,51,51
  fi
#
# Stitch the oblique volumes, using either a pre-existing alignment
# or the one calculated above.
#
oblique2stitched \
    --input $PWD/"$channel"_destriped_precomputed \
    --output $PWD/"$channel"_destriped_precomputed_stitched \
    --alignment $ALIGN_FILE \
    --levels 7 \
    --x-step-size "$X_STEP_SIZE" \
    --y-voxel-size "$Y_VOXEL_SIZE" \
    --z-offset "$Z_OFFSET" \
    --n-writers 11 \
    --n-workers 24
fi
#
# Convert the precomputed volume's level 1 blockfs to TIFFs
#
blockfs2jp2k \
    --input "$channel"_destriped_precomputed_stitched/1_1_1/precomputed.blockfs \
    --output-pattern "$channel"_destriped_stitched/img_%04d.jp2 \
    --n-workers $WORKERS \
    --psnr $PSNR
#
# Clean up by deleting all intermediate files
#
if [ $SINGLE_CHANNEL == 0 ]
then
  rm -r "$channel"_destriped_precomputed
fi

