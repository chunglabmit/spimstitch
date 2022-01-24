#!/bin/bash
#
# This pipeline processes three channels off of the oblique SPIM microscope
# Two of the channels are converted to JPEG 2000 in order to compress them
# for later processing and the third one is processed and aligned as a DANDI
# volume, but without destriping.
#
# Arguments:
# $1 - the subject name, e.g. MITU01
# $2 - the sample name, e.g. "50"
# $3 - the name of the channel to be processed, e.g. "NN"
# $4 - the relative path to that channel e.g. Ex_561_Em_2
# $5 - the relative path of the first channel to be converted to JPEG2000
# $6 - the relative path of the second channel to be converted to JPEG2000
#
# Environment variables:
# JP2K_BASE: the base directory for storing JPEG 2000 files, e.g.
#            /mnt/beegfs/Lee/data
#
# DANDI_ROOT: the base directory for the DANDI hierarchy, e.g.
#             /mnt/beegfs/Lee/dandi
#
# TEMPLATE: the JSON template for the sidecar
#
set -e
SUBJECT=$1
SAMPLE=$2
STAIN=$3
RELATIVE_DANDI_CHANNEL=$4
RELATIVE_FIRST_JP2K_CHANNEL=$5
RELATIVE_SECOND_JP2K_CHANNEL=$6
#################################
#
# Variables
#
#################################
if [ -z "$PSNR" ] # Signal to noise ratio for JPEG 2000 compression
then
  PSNR=80
fi
if [ -z "$N_WRITERS" ]
then
  N_WRITERS=18
fi
if [ -z "$N_DCIMG2OBLIQUE_WORKERS" ]
then
  N_DCIMG2OBLIQUE_WORKERS=12
fi
if [ -z "N_DCIMG2JP2K_WORKERS" ]
then
  N_DCIMG2JP2K_WORKERS=48
fi
if [ -z $DANDI_ROOT ]
then
  echo "Please define \$DANDI_ROOT"
  exit -1
fi
if [ -z $JP2K_BASE ]
then
  echo "Please define \$JP2K_BASE"
  exit -1
fi

#################################
#
# Directory checks
#
#################################

DANDI_CHANNEL=$(realpath "$RELATIVE_DANDI_CHANNEL")
ALL_DANDI_DCIMG="$DANDI_CHANNEL"/*/*/*.dcimg

if [ -z "$ALL_DANDI_DCIMG" ]
then
  echo "No DCIMG files on path $DANDI_CHANNEL"
  exit -1
fi

if [ -z "$RELATIVE_FIRST_JP2K_CHANNEL" ]
then
  echo "Not processing first JP2K channel"
else
  FIRST_JP2K_CHANNEL=$(realpath "$RELATIVE_FIRST_JP2K_CHANNEL")
  DCIMG2JP2K_FILES="$FIRST_JP2K_CHANNEL"/*/*/*.dcimg
  if [ -z "$DCIMG2JP2K_FILES" ]
  then
    echo "No DCIMG files on path $FIRST_JP2K_CHANNEL"
    exit -1
  fi
fi

if [ -z "$RELATIVE_SECOND_JP2K_CHANNEL" ]
then
  echo "Not processing second JP2K channel"
else
  SECOND_JP2K_CHANNEL=$(realpath "$RELATIVE_SECOND_JP2K_CHANNEL")
  SECOND_DCIMG2JP2K_FILES="$SECOND_JP2K_CHANNEL"/*/*/*.dcimg
  if [ -z "$SECOND_DCIMG2JP2K_FILES" ]
  then
    echo "No DCIMG files on path $FIRST_JP2K_CHANNEL"
    exit -1
  fi
  DCIMG2JP2K_FILES="$DCIMG2JP2K_FILES $SECOND_DCIMG2JP2K_FILES"
fi

if [ ! -d "$JP2K_BASE" ]
then
  echo "$JP2K_BASE is not a directory"
  exit -1
fi
if [ ! -d "$DANDI_ROOT" ]
then
  echo "$DANDI_ROOT is not a directory"
  exit -1
fi

if [ -z "$TEMPLATE" ]
then
  echo "\$TEMPLATE is not defined. Please set it to the location of the template file"
  exit -1
fi

if [ ! -f "$TEMPLATE" ]
then
  echo "$TEMPLATE (the template file) does not exist"
  exit -1
fi

METADATA_FILE=$(dirname "$DANDI_CHANNEL")/metadata.txt
if [ ! -f "$METADATA_FILE" ]
then
  echo "Missing file: $METADATA_FILE"
  exit -1
fi
X_STEP_SIZE=$(dandi-metadata get-x-step-size "$METADATA_FILE")
Y_VOXEL_SIZE=$(dandi-metadata get-y-voxel-size "$METADATA_FILE")

set -x

echo #######################################################
echo #
echo # SAMPLE:  $SAMPLE
echo # STAIN:   $STAIN
echo # CHANNEL: $DANDI_CHANNEL
if [ -n $DCIMG2JP2K_FILES ] then
  echo # JP2K1:  $FIRST_JP2K_CHANNEL
fi
if [ -n "$SECOND_JP2K_CHANNEL" ]
then
  echo # JP2K2:  $SECOND_JP2K_CHANNEL
fi
echo # JP2K_BASE: $JP2K_BASE
echo # DANDI_ROOT: $DANDI_ROOT
echo # ILLUM_CORR: $ILLUM_CORR
echo # X_STEP_SIZE: $X_STEP_SIZE
echo # Y_VOXEL_SIZE: $Y_VOXEL_SIZE
echo #
echo #######################################################

#############################################################
#
# Loop to process primary channel
#
#############################################################

CHUNK_NUMBER=0
for DCIMG_PATH in $ALL_DANDI_DCIMG
do
  CHUNK_NUMBER=$(( $CHUNK_NUMBER + 1 ))
  TARGET_NAME="$DANDI_ROOT"/$(dandi-metadata target-file \
    --subject $SUBJECT \
    --sample $SAMPLE \
    --source-path $(dirname "$DCIMG_PATH") \
    --stain $STAIN \
    --chunk $CHUNK_NUMBER )
  #
  # Write the NGFF volume
  #
  mkdir -p $(dirname "$TARGET_NAME")
  VOLUME_PATH="$TARGET_NAME".ngff
  SIDECAR_PATH="$TARGET_NAME".json
  dcimg2oblique \
    --n-writers $N_WRITERS \
    --n-workers $N_DCIMG2OBLIQUE_WORKERS \
    --rotate-90 3 \
    --flip-ud \
    --input "$DCIMG_PATH" \
    --output "$VOLUME_PATH" \
    --levels 7 \
    --y-voxel-size $Y_VOXEL_SIZE \
    --x-step-size $X_STEP_SIZE \
    --ngff
  #
  # Write the sidecar
  #
  	#
	# Write the sidecar.
	#
	dandi-metadata write-sidecar \
	  --template "$TEMPLATE" \
	  --metadata-file "$METADATA_FILE" \
	  --volume "$VOLUME_PATH" \
	  --volume-format ngff \
	  --stain "$STAIN" \
	  --y-voxel-size $Y_VOXEL_SIZE \
	  --dcimg-input "$DCIMG_PATH" \
	  --output "$SIDECAR_PATH" \
	  $ALL_DANDI_DCIMG
#
# At this point, the DCIMG file can be deleted
#
done
###########################################################################
#
# Alignment
#
###########################################################################

if [ $(echo $ALL_DANDI_DCIMG | wc -w) = "1" ]
then
  echo "No alignment needed"
else
  ALIGN_FILE="$DANDI_CHANNEL"-align.json
  ngff_wildcard=$(basename $(dandi-metadata target-file \
      --subject $SUBJECT \
      --sample $SAMPLE \
      --source-path $(dirname $RAW_PATH) \
      --stain $STAIN \
      --chunk "*" \
    ))".ngff"
  oblique-align \
  --ngff \
  --input $(dirname "$target_name") \
  --pattern "$ngff_wildcard" \
  --output $ALIGN_FILE \
  --voxel-size $Y_VOXEL_SIZE \
  --x-step-size $X_STEP_SIZE \
  --is-oblique \
  --n-cores "$N_WORKERS" \
  --sigma 10 \
  --sample-count 250 \
  --window-size 51,51,51 \
  --align-xz
fi

###########################################################################
#
# Write JPEG-2000 files
#
###########################################################################

for DCIMG_PATH in $DCIMG2JP2K_FILES
do
  Z=$(echo "$DCIMG_PATH" | cut -d. -f1)
  REST=$(dirname "$DCIMG_PATH")
  XY=$(basename "$REST")
  REST=$(dirname "$REST")
  X=$(basename "$REST")
  CHANNEL=$(basename $(dirname "$REST"))
  JP2K_PATH="$JP2K_BASE"/"$CHANNEL"/"$X"/"$XY"/"$Z"
  mkdir -p "$JP2K_PATH"

  dcimg2jp2 \
  --input $DCIMG_PATH \
  --output-pattern $JP2K_PATH/img_%05d.jp2 \
  --n-workers $N_DCIMG2JP2_WORKERS \
  --psnr $PSNR
done
