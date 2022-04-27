#!/bin/bash
#
# Arguments:
# dandi-pipeline.sh <directory> <stain> <sample>
#
# where <directory> is the path to the channel to be processed
#       <stain> is the name of the stain associated with the channel
#
# The pipeline should be run using the chunglab-stack
#
# Environment variables:
# $ILLUM_CORR - the path to the illumination correction function
# $N_WORKERS - the number of workers to use
# $DANDI_ROOT - root directory of dandiset
# $SUBJECT - the subject that was the source for the sample
# $PSNR - snr for compression (default = 80)
# $TEMPLATE - the JSON template for the sidecars
# $JP2K_ROOT - place to put JPEG 2000 files.
# $ALIGN_FILE - the output of the oblique-align command from a previous channel or not present to calculate it
# $ALIGN_XZ - if defined, align in the X and Z direction as well
#
RAW_PATH=$1
if [ ! -d "$RAW_PATH" ];
then
  echo "$RAW_PATH does not exist. Exiting."
  exit 1
fi
if [ -z "$SUBJECT" ];
then
  echo "Please define the SUBJECT environment variable"
  exit 1
fi
CHANNEL=$(basename "$RAW_PATH")

STAIN=$2
SAMPLE=$3
if [ -z "$PSNR" ]; then export PSNR=80; fi
if [ -z "$TEMPLATE" ]; then
  TEMPLATE=/mnt/beegfs/Lee/dandi/template.json
fi
METADATA_FILE=$(dirname "$RAW_PATH")/metadata.txt
X_STEP_SIZE=$(dandi-metadata get-x-step-size "$METADATA_FILE")
Y_VOXEL_SIZE=$(dandi-metadata get-y-voxel-size "$METADATA_FILE")
NEGATIVE_Y=$(dandi-metadata get-negative-y "$METADATA_FILE")
if [ $(dandi-metadata get-flip-y "$METADATA_FILE" "$CHANNEL") == "flip-y" ];
then
  FLIP_Y_SWITCH="--flip-ud"
else
  FLIP_Y_SWITCH=""
fi

if [ -z "$ILLUM_CORR" ]; then
  ILLUM_CORR=/media/share10/lee/illum/ospim1-2021-03-09.tiff
fi
if [ -z "$N_WORKERS" ]; then
  N_WORKERS=48
fi
if [ -z "$JP2K_ROOT" ]; then
  JP2K_ROOT="$RAW_PATH"_jp2k
fi
if [ -z "$SUBJECT" ]; then
  echo "SUBJECT undefined"
  exit 1
fi
#
#----- Run parameters
#
set -e
echo "--------- Run parameters --------"
echo "Path:          $RAW_PATH"
echo "Subject:       $SUBJECT"
echo "Stain:         $STAIN"
echo "Sample:        $SAMPLE"
echo "Metadata file: $METADATA_FILE"
echo "X_STEP_SIZE:   $X_STEP_SIZE"
echo "Y_VOXEL_SIZE:  $Y_VOXEL_SIZE"
echo "ILLUM_CORR:    $ILLUM_CORR"
echo "N_WORKERS:     $N_WORKERS"
echo "DANDI_ROOT:    $DANDI_ROOT"
echo "JP2K_ROOT      $JP2K_ROOT"
echo "FLIP_Y_SWITCH" $FLIP_Y_SWITCH
echo "ALIGN_XZ"      $ALIGN_XZ
echo "--------------------------------"
#
# Spirious warning from undeleted shared memory in subprocesses
#
PYTHONWARNINGS=ignore
#
# Increase the number of simultaneous open files
#
ulimit -n 65535
# Loop over each .dcimg file
#
ALL_DCIMGS=$(find $RAW_PATH -wholename "**/*.dcimg")
ALL_DCIMGS=$(dandi-metadata order-dcimg-files $ALL_DCIMGS)
STACK_COUNT=$( echo "$ALL_DCIMGS"| wc -w )

CHUNK_NUMBER=0
for DCIMG_PATH in $ALL_DCIMGS;
do
  CHUNK_NUMBER=$(($CHUNK_NUMBER + 1))
  dcimg_filename=$(basename "$DCIMG_PATH")
  z=$(echo "$dcimg" | cut -d. -f1)
  rest=$(dirname "$DCIMG_PATH")
  xy=`basename $rest`
  rest=`dirname $rest`
  x=`basename $rest`
  jp2k_path="$JP2K_ROOT"/"$x"/"$xy"/"$z"
  mkdir -p "$jp2k_path"
  #
  # Write the JPEG files
  #
  dcimg2jp2 \
  --input $DCIMG_PATH \
  --output-pattern "$jp2k_path/img_%05d.jp2" \
  --n-workers $N_WORKERS \
  --psnr $PSNR

  target_name="$DANDI_ROOT"/$(dandi-metadata target-file \
      --subject $SUBJECT \
      --sample $SAMPLE \
      --source-path "$(dirname "$RAW_PATH")" \
      --stain $STAIN \
      --chunk $CHUNK_NUMBER )
  #
  # Write the NGFF volume
  #
  mkdir -p $(dirname "$target_name")
  volume_path="$target_name".ngff
  sidecar_path="$target_name".json
  transform_path=${target_name::-4}transforms.json
  dcimg2oblique \
    --n-writers 11 \
    --n-workers $N_WORKERS \
    --rotate-90 3 \
    $FLIP_Y_SWITCH \
    --input "$jp2k_path"/"img_*.jp2" \
    --output "$volume_path" \
     --destripe-method wavelet --sigma1 128 --sigma2 512 --wavelet db5 --crossover 10 \
     --flat $ILLUM_CORR --dark 100 \
    --levels 7 \
    --y-voxel-size $Y_VOXEL_SIZE \
    --x-step-size $X_STEP_SIZE \
    --jp2k \
    --ngff
	#
	# Write the sidecar.
	#
	dandi-metadata write-sidecar \
	  --template "$TEMPLATE" \
	  --metadata-file "$METADATA_FILE" \
	  --volume "$volume_path" \
	  --volume-format ngff \
	  --stain "$STAIN" \
	  --y-voxel-size $Y_VOXEL_SIZE \
	  --dcimg-input "$DCIMG_PATH" \
	  --output "$sidecar_path" \
	  $ALL_DCIMGS
	#
	# At this point, the dcimg file could be deleted... not yet though
	#
done
if [ $STACK_COUNT == "1" ]; then
  echo "Single stack"
else
  if [ -z $ALIGN_FILE ]; then
    ALIGN_FILE="$RAW_PATH"-align.json
  fi
  if [ ! -f $ALIGN_FILE ]; then
    if [ -d "$DONT_ALIGN_XZ" ];
    then
      ALIGN_EXTRAS=--align-xz
    fi
  if [ -v NEGATIVE_Y ];
  then
    export ALIGN_EXTRAS=$ALIGN_EXTRAS" --negative-y"
  fi
  ngff_wildcard=$(basename $(dandi-metadata target-file \
        --subject $SUBJECT \
        --sample $SAMPLE \
        --source-path $(dirname "$RAW_PATH") \
        --stain $STAIN \
        --chunk "*" \
    ))".ngff"
    oblique-align \
      --ngff \
      --input "$(dirname "$target_name")" \
      --pattern "$ngff_wildcard" \
      --output "$ALIGN_FILE" \
      --voxel-size $Y_VOXEL_SIZE \
      --x-step-size $X_STEP_SIZE \
      --is-oblique \
      --n-cores "$N_WORKERS" \
      --sigma 10 \
      --sample-count 250 \
      --window-size 51,51,51 \
      --report "${ALIGN_FILE%.*}.pdf" \
      $ALIGN_EXTRAS
  fi
  sidecar_wildcard="$(basename $(dandi-metadata target-file \
    --subject $SUBJECT \
    --sample $SAMPLE \
    --source-path $(dirname "$RAW_PATH") \
    --stain $STAIN \
    --chunk "*" ))".json
  ALL_SIDECAR_FILES=$(find $(dirname "$target_name") -maxdepth 1 -name "$sidecar_wildcard")
  dandi-metadata rewrite-transforms \
      --align-file "$ALIGN_FILE" \
      --y-voxel-size $Y_VOXEL_SIZE \
      $ALL_SIDECAR_FILES
fi